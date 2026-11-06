"""Experimental differentiable Sérsic/empirical-PSF FFT renderer.

Finite-domain quadrature is an approximation. This backend is NOT enabled by
production configuration. Validate its entire intended morphology domain and
noise-weighted image/gradient error against GalSim before inference.
"""
import numpy as np
from ..cache import ExactLRU
from ..rendering import GalSimRenderer
from ..spatial import PointSource


class JAXRenderer:
    def __init__(self,*,oversampling=4,padding=2,experimental=False,cache_bytes=512*1024**2):
        if not experimental:raise ValueError('JAX Sérsic rendering is experimental and requires explicit accuracy validation')
        if oversampling<2 or int(oversampling)!=oversampling or oversampling%2:
            raise ValueError('Use an even oversampling >=2 for aligned detector-grid samples')
        if padding<2:raise ValueError('Use padding >=2 to assess convolution tails')
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import gammainc,gammaln
        jax.config.update('jax_enable_x64',True)
        self.jax,self.jnp=jax,jnp
        self.oversampling=int(oversampling);self.padding=float(padding)
        self.reference=GalSimRenderer();self._prepared=ExactLRU(cache_bytes)
        self.metrics=dict(psf_preparations=0,batch_calls=0)

        @jax.custom_jvp
        def bn(n):
            def step(_,state):
                lo,hi=state;mid=(lo+hi)/2
                lower=gammainc(2*n,mid)<.5
                return jnp.where(lower,mid,lo),jnp.where(lower,hi,mid)
            lo,hi=jax.lax.fori_loop(0,64,step,(jnp.array(0.),2*n+10.))
            return (lo+hi)/2
        @bn.defjvp
        def bn_jvp(primals,tangents):
            n,=primals;dn,=tangents;b=bn(n)
            da=jax.grad(lambda a:gammainc(a,b))(2*n)
            db=jnp.exp((2*n-1)*jnp.log(b)-b-gammaln(2*n))
            return b,-2*da/db*dn
        self.bn=bn

        def render(parameters,points,psf_fft,x,y,kx,ky,matrix,offset,shift,indices_y,indices_x):
            inverse=jnp.linalg.inv(matrix);det=jnp.abs(jnp.linalg.det(matrix))
            ny,nx=len(indices_y),len(indices_x)
            center=jnp.array([(nx-1)/2,(ny-1)/2])
            def component(p,point):
                cx,cy=(matrix@p[:2]+offset+shift-center)
                dx=x-cx;dy=y-cy
                rx=inverse[0,0]*dx+inverse[0,1]*dy
                ry=inverse[1,0]*dx+inverse[1,1]*dy
                angle=p[5]*jnp.pi/180
                u=rx*jnp.cos(angle)+ry*jnp.sin(angle)
                v=-rx*jnp.sin(angle)+ry*jnp.cos(angle)
                radius=jnp.sqrt(jnp.maximum(u*u+(v/p[4])**2,1e-28))
                b=bn(p[3])
                logamp=2*p[3]*jnp.log(b)-jnp.log(2*jnp.pi*p[3])-gammaln(2*p[3])-2*jnp.log(p[2])-jnp.log(p[4])-jnp.log(det)
                intrinsic=jnp.exp(logamp-b*(radius/p[2])**(1/p[3]))/self.oversampling**2
                transform=jnp.fft.fft2(jnp.fft.ifftshift(intrinsic))
                transform=jnp.where(point,jnp.exp(-1j*(kx*cx+ky*cy)),transform)
                return transform*psf_fft
            transformed=jax.vmap(component)(parameters,points)
            convolved=jnp.fft.fftshift(jnp.fft.ifft2(transformed,axes=(-2,-1)),axes=(-2,-1)).real
            return convolved[:,indices_y[:,None],indices_x[None,:]]
        self._render=jax.jit(render)

    def cache_token(self):return ('experimental_jax_fft',self.oversampling,self.padding)

    def prepare(self,observation):
        from scipy.fft import next_fast_len
        key=(observation.name,observation.psf.cache_token(),observation.data.shape,self.oversampling,self.padding)
        cached=self._prepared.get(key)
        if cached is not None:return cached
        if not observation.psf.pixel_integrated:raise ValueError('Experimental backend currently requires an already pixel-integrated PSF')
        ny,nx=observation.data.shape;s=self.oversampling
        fy=next_fast_len(int(np.ceil(self.padding*ny)))*s
        fx=next_fast_len(int(np.ceil(self.padding*nx)))*s
        # GalSim supplies the declared-center Lanczos interpolant. The kernel
        # already contains the detector response; no pixel is convolved again.
        obj=self.reference._psf(observation)
        kernel=np.asarray(obj.drawImage(nx=fx,ny=fy,scale=1./s,offset=self.reference.gs.PositionD(.5,.5),
            method='no_pixel',dtype=np.float64).array)*s*s
        x=(np.arange(fx)-fx//2)/s;y=(np.arange(fy)-fy//2)/s
        kx=2*np.pi*np.fft.fftfreq(fx,d=1./s);ky=2*np.pi*np.fft.fftfreq(fy,d=1./s)
        iy=(fy//2-(ny-1)*s//2+np.arange(ny)*s).astype(int)
        ix=(fx//2-(nx-1)*s//2+np.arange(nx)*s).astype(int)
        cached=dict(psf_fft=self.jnp.asarray(np.fft.fft2(np.fft.ifftshift(kernel))),
            x=self.jnp.asarray(x[None,:]),y=self.jnp.asarray(y[:,None]),
            kx=self.jnp.asarray(kx[None,:]),ky=self.jnp.asarray(ky[:,None]),
            indices_y=self.jnp.asarray(iy),indices_x=self.jnp.asarray(ix))
        self._prepared[key]=cached;self.metrics['psf_preparations']+=1
        return cached

    def render_jax(self,parameters,points,observation,shift=(0.,0.)):
        args=self.prepare(observation);self.metrics['batch_calls']+=1
        return self._render(parameters,points,matrix=self.jnp.asarray(observation.matrix),
            offset=self.jnp.asarray(observation.offset),shift=self.jnp.asarray(shift),**args)

    def render_batch(self,components,morphologies,observation,shift=(0.,0.)):
        rows=[];points=[]
        for c,m in zip(components,morphologies):
            rows.append([m['x'],m['y'],m.get('re',1.),m.get('n',1.),m.get('q',1.),m.get('pa',0.)])
            points.append(isinstance(c,PointSource))
        return self.render_jax(self.jnp.asarray(rows),self.jnp.asarray(points),observation,shift)

    def render(self,component,morphology,observation,shift=(0.,0.)):
        return np.asarray(self.render_batch([component],[morphology],observation,shift)[0])

"""Explicit synthetic models for sampler validation, never target defaults."""
import numpy as np
from scipy.special import erf
from chromorph import Scene,Observation,EmpiricalPSF,PointSource,Sersic,FreeBandSED,Parameter,SEDContext


class ColorSED:
    """Analytic two-band test SED; not a stellar-population approximation."""
    def __init__(self,mass=0.,color=.3):
        self.parameters={'log_mass':Parameter(mass,-.6,.6),'color':Parameter(color,-1.2,1.2)}
    def register(self,registry,name,bands):
        for k,p in self.parameters.items():registry.add(name+'.sed.'+k,p)
    def fluxes(self,values,name,observations,context):
        m=10**values[name+'.sed.log_mass'];c=values[name+'.sed.color']
        return {o.band:m*np.exp(c if i==0 else -c) for i,o in enumerate(observations)}
    def describe(self):return dict(model='analytic_independent_color_test')


class IntegratedGaussianRenderer:
    """Exact synthetic Gaussian convolution; independent test-only renderer."""
    def render(self,component,morphology,observation,shift=(0,0)):
        from chromorph.spatial import Sersic
        xy=observation.matrix@np.array([morphology['x'],morphology['y']])+observation.offset+shift
        sigma=1.
        if isinstance(component,Sersic):
            if morphology['n']!=.5 or morphology['q']!=1:raise ValueError('Analytic test renderer requires its declared Gaussian family')
            sigma=np.hypot(sigma,morphology['re']/np.sqrt(2*np.log(2)))
        yy,xx=np.indices(observation.data.shape)
        def integral(grid,center):return (erf((grid-center+.5)/(np.sqrt(2)*sigma))-erf((grid-center-.5)/(np.sqrt(2)*sigma)))/2
        return integral(xx,xy[0])*integral(yy,xy[1])


def synthetic_case(seed=101,*,overlap=True,full_morphology=False):
    yy,xx=np.indices((19,19));psf=np.exp(-((xx-9)**2+(yy-9)**2)/2);psf/=psf.sum()
    obs=[Observation(b,b,np.zeros((19,19)),np.full((19,19),.025),EmpiricalPSF(psf,.04),.04,1.) for b in ['blue','red']]
    if full_morphology:
        from chromorph.rendering import GalSimRenderer
        renderer=GalSimRenderer()
    else:renderer=IntegratedGaussianRenderer()
    scene=Scene(obs,renderer=renderer)
    scene.add_component(PointSource('nucleus',FreeBandSED({'blue':.7,'red':1.1}),parameters={k:Parameter(9.,vary=False) for k in ['x','y']}))
    centers=[8.,10.] if overlap else [5.,13.]
    for j,x in enumerate(centers):
        scene.add_component(Sersic('clump'+str(j+1),ColorSED(-.05*j,.4 if j==0 else -.4),parameters={
            'x':Parameter(x,3.,15.),'y':Parameter(9.,6.,12.,vary=full_morphology),
            're':Parameter(1.2+.2*j,.5,3.),'n':Parameter(.5,.3,1.3,vary=full_morphology),
            'q':Parameter(.8 if full_morphology else 1.,.4,1.,vary=full_morphology),
            'pa':Parameter(20.*j,-180.,180.,vary=full_morphology)}))
    truth=scene.registry.unpack();images=scene.predict(truth)[0];rng=np.random.default_rng(seed)
    for o in obs:o.update(data=images[o.name]+rng.normal(0,o.sigma,o.data.shape))
    return scene,truth

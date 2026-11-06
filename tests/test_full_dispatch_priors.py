import numpy as np
import pytest
from chromorph import Scene,PointSource,FreeBandSED,Parameter
from chromorph.direct import FullPixelLikelihood,AutoGaussianLikelihood
from chromorph.priors import Prior,prior_for
from chromorph.noise import DenseCovariance
from conftest import observations,clump


def build(renderer,n=4):
 s=Scene(observations(('b','r'),size=19),renderer=renderer,cache_bytes=0)
 s.add_component(PointSource('nucleus',FreeBandSED(),x=9.,y=9.))
 for j in range(n):s.add_component(clump('c'+str(j),x=4.+3*j,y=8.,moving=True))
 return s


def test_full_path_requires_neither_gram_nor_spatial_cache(renderer):
 s=build(renderer);v=s.registry.unpack();engine=FullPixelLikelihood(s)
 for i in range(5):
  for c in s.components:
   for k in c.parameters:
    if k in {'x','y'}:v[c.name+'.morphology.'+k]+=.03
    elif k=='re':v[c.name+'.morphology.'+k]*=1.01
  actual=engine.loglikelihood(v)
  assert s._compressed is None and len(s._basis_cache)==0
  expected=s.pixel_loglikelihood(v)
  assert actual==pytest.approx(expected,rel=1e-12,abs=1e-8)
 assert engine.metrics['render_calls']==5*len(s.observations)*len(s.components)


def test_dispatch_and_incremental_statistics(renderer):
 s=Scene(observations(size=19),renderer=renderer)
 s.add_component(clump('a',moving=True));s.add_component(clump('b',x=14.,moving=True))
 engine=AutoGaussianLikelihood(s);v=s.registry.unpack()
 first=engine.loglikelihood(v);count=s.metrics['render_calls']
 v['a.flux.blue']*=1.2;engine.loglikelihood(v)
 assert s.metrics['render_calls']==count
 stat=s.compressed.statistics(v);before={b:(x.G.copy(),x.h.copy()) for b,x in stat.items()}
 v['a.morphology.x']+=.1
 stat=s.compressed.statistics(v)
 for b,x in stat.items():
  np.testing.assert_array_equal(x.G[1:,1:],before[b][0][1:,1:])
  np.testing.assert_array_equal(x.h[1:],before[b][1][1:])
 assert s.compressed.metrics['rank_updates']==2
 for i in range(15):
  for c in s.components:v[c.name+'.morphology.x']+=.02
  assert engine.loglikelihood(v)==pytest.approx(s.pixel_loglikelihood(v),rel=1e-12,abs=1e-8)
 assert engine.metrics['all_components_changed']>=1


@pytest.mark.parametrize('prior',[Prior('normal',1.,.4),Prior('lognormal',-.2,.5),Prior('uniform',lower=.1,upper=3.)])
def test_normalized_priors_and_inverse_transform(prior):
 from scipy.integrate import quad
 p=Parameter(1.,.05,4.,prior=prior);d=prior_for(p)
 assert quad(lambda x:np.exp(d.logpdf(x)),d.lower,d.upper,epsabs=1e-9,points=[d.mode])[0]==pytest.approx(1.,rel=1e-7)
 x=d.ppf(np.linspace(.001,.999,100));assert np.all((x>=d.lower)&(x<=d.upper))
 for x in [float(d.ppf(.2)),float(d.ppf(.7))]:
  assert d.residual(x)**2==pytest.approx(2*(d.logpdf(d.mode)-d.logpdf(x)),rel=1e-10,abs=1e-12)


def test_explicit_independent_nuclear_amplitude_priors_and_cost_groups(renderer):
 s=Scene(observations(),renderer=renderer,allow_nuclear_flux_priors=True)
 s.add_component(PointSource('nucleus',FreeBandSED(parameters={b:Parameter(1.,0.,10.) for b in s.bands}),parameters={'x':Parameter(9.,vary=False),'y':Parameter(9.,vary=False)}))
 s.add_component(clump('one',moving=True,sed=FreeBandSED(parameters={b:Parameter(1.,0.,10.) for b in s.bands})))
 groups=s.registry.cost_groups()
 assert 'nucleus.flux.blue' in groups['fast'] and 'one.morphology.x' in groups['slow']
 assert np.isfinite(s.registry.logprior(s.registry.unpack()))
 assert not s.registry.ties

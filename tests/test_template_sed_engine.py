import numpy as np
import pytest
from dataclasses import asdict
from chromorph import Scene,PointSource,FreeBandSED,Parameter,SEDContext
from chromorph.templates import BandFluxLibrary,SSPBandLibrary,PrecomputedFSPSSED,observational_signature
from chromorph.priors import Prior,prior_for,CubePriorTransform
from conftest import observations,clump


def metadata(obs,context,sfh='ssp'):
 return dict(bands=[o.band for o in obs],signature=observational_signature(obs,context),
    fixed_shape=dict(metallicity=-.3,av=.2),reference=dict(sfh=sfh,nebular=False,dust=True,configuration=dict(dust_type=2)),validation=None)


def test_ssp_age_integral_uses_linear_age_and_formed_mass(tmp_path):
 obs=observations();ctx=SEDContext(distance_mpc=10);ages=np.array([.001,.1,1.,10.])
 flux=np.array([[1+t,2*(1+t)] for t in ages])
 library=SSPBandLibrary({'log_age':np.log10(ages)},flux,metadata(obs,ctx))
 shape=dict(age=.35,metallicity=-.3,av=.2)
 expected=1+shape['age']/2+.5*ages[0]**2/shape['age']
 np.testing.assert_allclose(library.predict(shape),[expected,2*expected],rtol=1e-14)
 library.save(tmp_path/'library.npz');restored=BandFluxLibrary.load(tmp_path/'library.npz')
 np.testing.assert_array_equal(restored.predict(shape),library.predict(shape))


def test_template_component_independence_and_mass_scaling():
 obs=observations();ctx=SEDContext(distance_mpc=10)
 library=BandFluxLibrary({'log_age':[-2.,0.]},np.array([[1.,3.],[2.,.5]]),metadata(obs,ctx,'constant'))
 parameters=dict(metallicity=Parameter(-.3,vary=False),av=Parameter(.2,vary=False))
 with Scene(obs,context=ctx) as s:
  a=PrecomputedFSPSSED(library,parameters=parameters,experimental=True,fallback='raise')
  b=PrecomputedFSPSSED(library,parameters=parameters,experimental=True,fallback='raise')
  s.add_component(clump('a',sed=a));s.add_component(clump('b',sed=b))
  v=s.registry.unpack();v['a.sed.age']=.1;v['b.sed.age']=.8
  fa=a.fluxes(v,'a',obs,ctx);fb=b.fluxes(v,'b',obs,ctx)
  v['a.sed.age']=.2;assert a.fluxes(v,'a',obs,ctx)!=fa and b.fluxes(v,'b',obs,ctx)==fb
  old=a.fluxes(v,'a',obs,ctx);v['a.sed.log_mass']+=1
  np.testing.assert_allclose(list(a.fluxes(v,'a',obs,ctx).values()),10*np.array(list(old.values())))
  assert a.metrics['fsps_calls']==b.metrics['fsps_calls']==0
  with pytest.raises(ValueError,match='context changed'):a.unit_fluxes(v,'a',obs,SEDContext(distance_mpc=20))
  v['a.sed.age']=20
  with pytest.raises(ValueError):a.unit_fluxes(v,'a',obs,ctx)


def test_unvalidated_templates_cannot_silently_enter_inference():
 obs=observations();ctx=SEDContext(distance_mpc=10)
 library=BandFluxLibrary({'log_age':[-2.,0.]},np.ones((2,2)),metadata(obs,ctx,'constant'))
 sed=PrecomputedFSPSSED(library,fallback='raise')
 v={'a.sed.'+k:p.value for k,p in sed.parameters.items()}
 with pytest.raises(ValueError,match='accuracy gate'):sed.unit_fluxes(v,'a',obs,ctx)


def test_ml_nuclear_amplitudes_are_not_limited_by_bayesian_prior(renderer):
 obs=observations(('blue',),size=19)
 s=Scene(obs,renderer=renderer)
 s.add_component(PointSource('nucleus',FreeBandSED(parameters={'blue':Parameter(1.,0.,np.inf,prior=Prior('uniform',lower=0,upper=2))}),
     parameters={k:Parameter(9.,vary=False) for k in ['x','y']}))
 v=s.registry.unpack();v['nucleus.flux.blue']=8.;obs[0].data[:]=s.predict(v)[0]['blue']
 result=s.fit(objective='ml',multistart=1,estimate_uncertainty=False)
 assert result.values['nucleus.flux.blue']==pytest.approx(8.,rel=1e-9)
 assert result.objective_mode=='ml' and s.registry.parameters['nucleus.flux.blue'].prior is not None


def test_vectorized_priors_equal_scalar_transform():
 from chromorph.parameters import Registry
 r=Registry()
 specs=[Prior('normal',1.,.3),Prior('lognormal',-.5,.4),Prior('uniform',lower=.1,upper=10.)]
 for j,spec in enumerate(specs):r.add('a'+str(j),Parameter(1.,.001,10,prior=spec))
 transform=CubePriorTransform(r,r.names())
 for cube in [.001,.5,.999]:
  actual=transform(np.full(3,cube));expected=[prior_for(p).ppf(cube) for p in r.parameters.values()]
  np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)

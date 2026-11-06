"""Independent full-pixel checks for layouts and globally profiled derivatives."""
from dataclasses import replace
import pickle
import numpy as np
import pytest
from scipy.optimize import lsq_linear
from chromorph import Scene,PointSource,FreeBandSED,Parameter,Prior
from chromorph.objectives import ProfiledObjective
from chromorph.noise import whitening_for
from conftest import observations,clump


def independent_profile(scene,values,names):
    """Fresh pixel design and bounded solve, without QR or evaluator caches."""
    base=dict(values)
    for name in names:base[name]=0.
    matrices=[];targets=[];basis={};whiteners={}
    for o in scene.observations:
        shift=np.array([values['observation.'+o.name+'.'+a] for a in ['dx','dy']])
        columns={c.name:scene.renderer.render(c,scene.morphology(c,values),o,shift) for c in scene.components}
        basis[o.name]=columns;w=whitening_for(o);whiteners[o.name]=w
        rate=np.exp(values['observation.'+o.name+'.log_calibration'])/o.ujy_per_rate
        baseline=np.full(o.data.shape,values['observation.'+o.name+'.background'])
        for c in scene.components:baseline+=base[c.name+'.flux.'+o.band]*rate*columns[c.name]
        matrix=np.zeros((o.mask.sum(),len(names)))
        for j,name in enumerate(names):
            component,_,band=name.split('.')
            if band==o.band:matrix[:,j]=w.whiten((rate*columns[component])[o.mask])
        matrices.append(matrix);targets.append(w.whiten((o.data-baseline)[o.mask]))
    lower=[scene.registry.parameters[n].lower for n in names];upper=[scene.registry.parameters[n].upper for n in names]
    answer=lsq_linear(np.vstack(matrices),np.concatenate(targets),bounds=(lower,upper),method='bvls',tol=1e-12)
    updated=dict(values);updated.update(zip(names,answer.x));residual=[]
    for o in scene.observations:
        model=np.full(o.data.shape,updated['observation.'+o.name+'.background'])
        rate=np.exp(updated['observation.'+o.name+'.log_calibration'])/o.ujy_per_rate
        for c in scene.components:model+=updated[c.name+'.flux.'+o.band]*rate*basis[o.name][c.name]
        residual.append(whiteners[o.name].whiten((o.data-model)[o.mask]))
    return np.r_[np.concatenate(residual),scene.registry.prior_residuals(updated)]


@pytest.mark.parametrize('tied',[False,True])
def test_profiled_jacobian_matches_independent_full_pixel_solver(renderer,tied):
    with Scene(observations(size=17),renderer=renderer) as scene:
        scene.add_component(clump('one',x=6,y=8,moving=True))
        scene.add_component(clump('two',x=9,y=8,moving=True))
        if tied:scene.tie('two.morphology.y','one.morphology.y')
        values=scene.registry.unpack();values.update({'one.flux.blue':2.,'two.flux.blue':-.2,
            'one.flux.red':.4,'two.flux.red':2.})
        # Build a negative-amplitude injection directly to force an active bound;
        # fitted physical amplitudes remain nonnegative.
        for o in scene.observations:
            model=sum(values[c.name+'.flux.'+o.band]*renderer.render(c,scene.morphology(c,values),o) for c in scene.components)
            o.data[:]=model
        objective=ProfiledObjective(scene)
        try:
            vector=scene.registry.initial(objective.names);base=objective.residual(vector)
            jac=objective.jacobian(vector);points,steps=objective.difference_points(vector)
            expected_base=independent_profile(scene,scene.registry.unpack(vector,objective.names),objective.projection.names)
            expected=np.column_stack([(independent_profile(scene,scene.registry.unpack(p,objective.names),objective.projection.names)-expected_base)/h
                                      for p,h in zip(points,steps)])
            np.testing.assert_allclose(base,expected_base,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(jac,expected,rtol=1e-6,atol=1e-6)
            assert objective.metrics['jacobians']==1
            assert objective.projection.metrics['solves']==1+len(vector)
        finally:objective.close()


def test_active_set_transition_uses_complete_profiled_residual(renderer):
    with Scene(observations(('blue',),size=17),renderer=renderer) as scene:
        c=clump('one',x=8,y=8,moving=True)
        for key in ['y','re']:c.parameters[key]=replace(c.parameters[key],vary=False)
        scene.add_component(c);v=scene.registry.unpack();o=scene.observations[0]
        unit=renderer.render(c,scene.morphology(c,v),o)
        _,xx=np.indices(o.data.shape);o.data[:]=(xx-8)*unit
        objective=ProfiledObjective(scene)
        vector=scene.registry.initial(objective.names);base=objective.residual(vector)
        point,step=objective.difference_points(vector)
        jac=objective.jacobian(vector)
        ref0=independent_profile(scene,v,objective.projection.names)
        ref1=independent_profile(scene,scene.registry.unpack(point[0],objective.names),objective.projection.names)
        np.testing.assert_allclose(jac[:,0],(ref1-ref0)/step[0],rtol=1e-6,atol=1e-6)
        assert not np.array_equal(base,ref1)
        objective.close()


def test_layout_revisions_ties_pickle_and_nonprojectable_priors(renderer):
    with Scene(observations(),renderer=renderer) as scene:
        scene.add_component(clump('one',moving=True));scene.add_component(clump('two',x=10,moving=True))
        before=scene.registry.layout;order=scene.registry.names()
        assert scene.registry.layout is before
        scene.tie('two.morphology.x','one.morphology.x')
        assert scene.registry.layout is not before
        values=scene.registry.unpack();assert values['two.morphology.x']==values['one.morphology.x']
        name='one.flux.blue';p=scene.registry.parameters[name]
        scene.registry.parameters[name]=replace(p,prior=Prior('uniform',lower=0,upper=10))
        objective=ProfiledObjective(scene)
        assert name not in objective.projection.names and name in objective.names
        restored=pickle.loads(pickle.dumps(scene.registry))
        assert restored.names()==scene.registry.names()
        assert restored.unpack()==scene.registry.unpack()
        objective.close()


def test_parallel_profiled_stellar_masses_match_serial_without_nested_fsps(renderer):
    from test_linear_projection import AnalyticStellarSED
    with Scene(observations(size=17),renderer=renderer) as scene:
        scene.add_component(clump('one',x=5,y=8,moving=True,sed=AnalyticStellarSED()))
        scene.add_component(clump('two',x=11,y=8,moving=True,sed=AnalyticStellarSED()))
        values=scene.registry.unpack();models=scene.predict(values)[0]
        for o in scene.observations:o.data[:]=models[o.name]
        serial=ProfiledObjective(scene);parallel=ProfiledObjective(scene,workers=2)
        try:
            vector=scene.registry.initial(serial.names)
            np.testing.assert_allclose(parallel.jacobian(vector),serial.jacobian(vector),rtol=1e-6,atol=1e-6)
            assert parallel.workers.metrics['tasks']==len(vector)
            assert parallel.workers.metrics['linear_solves']==len(vector)
        finally:parallel.close();serial.close()

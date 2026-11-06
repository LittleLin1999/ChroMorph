"""Independent spectral states; flux/mass-only proposals need no synthesis."""
from concurrent.futures import ThreadPoolExecutor
from .sed import FSPSStellarSED


class SpectralEngine:
    def __init__(self,scene,*,workers=1):
        self.scene=scene;self.workers=int(workers);self._unit={};self.pool=None;self.backends={};self._sources={}
        if self.workers<1:raise ValueError('Spectral workers must be positive')
        self.metrics=dict(shape_evaluations=0,parallel_batches=0)

    def fluxes(self,values):
        scene=self.scene;pending=[];answers={}
        throughputs=tuple((o.band,getattr(o.throughput,'fingerprint',None) or id(o.throughput)) for o in scene.observations)
        for component in scene.components:
            sed=component.sed
            if isinstance(sed,FSPSStellarSED) and type(sed).fluxes is FSPSStellarSED.fluxes:
                token=(id(sed),tuple(sorted(sed.configuration.items())),sed.sfh,sed.dust,sed.nebular,id(getattr(sed,'library',None)),getattr(sed,'experimental',None),getattr(sed,'fallback',None))
                if self._sources.get(component.name)!=token:
                    old=self.backends.pop(component.name,None)
                    if old is not None:old.close()
                    self.backends[component.name]=sed.prepared_copy();self._sources[component.name]=token
                sed=self.backends[component.name]
                if hasattr(sed,'library'):
                    for o in scene.observations:o.ensure_frozen()
                    sed._domain_authorized=sed.library.authorizes(values,component.name,scene.observations)
                key=(id(sed),sed._shape_key(values,component.name,scene.context),throughputs,getattr(sed,'_domain_authorized',None))
                previous=self._unit.get(component.name)
                if previous is None or previous[0]!=key:pending.append((component,key))
            else:answers[component.name]=sed.fluxes(values,component.name,scene.observations,scene.context)
        def evaluate(item):
            c,key=item
            return c.name,key,self.backends[c.name].unit_fluxes(values,c.name,scene.observations,scene.context)
        if len(pending)>1 and self.workers>1:
            if self.pool is None:self.pool=ThreadPoolExecutor(max_workers=self.workers)
            evaluated=self.pool.map(evaluate,pending);self.metrics['parallel_batches']+=1
        else:evaluated=map(evaluate,pending)
        for name,key,unit in evaluated:self._unit[name]=(key,dict(unit));self.metrics['shape_evaluations']+=1
        for c in scene.components:
            if isinstance(c.sed,FSPSStellarSED) and type(c.sed).fluxes is FSPSStellarSED.fluxes:answers[c.name]={b:10**values[c.name+'.sed.log_mass']*f for b,f in self._unit[c.name][1].items()}
        return answers

    def close(self):
        if self.pool is not None:self.pool.shutdown(wait=True);self.pool=None
        for backend in self.backends.values():backend.close()
        self.backends.clear();self._sources.clear();self._unit.clear()

"""Uniform actor admission and unchanged native missing-history masks."""
import json
from pathlib import Path

def install(job, output):
    from tbsim.envs.env_trajdata import EnvUnifiedSimulation
    import ccdiff.evaluation.selection_manifest as manifest
    original_admission=EnvUnifiedSimulation._disable_offroad_agents
    record=json.loads(Path(job['manifest']).read_text())
    required=set(job.get('candidate_ids',[record['adversary_vehicle_id']]))
    def admission(self,scene,*a,**kw):
        available=list(scene.agents)
        original_admission(self,scene,*a,**kw)
        native={x.name for x in scene.agents}
        scene.agents=[x for x in available if x.name in native|required]
        assert required <= {x.name for x in scene.agents}
        Path(output,'population.json').write_text(json.dumps(dict(native=sorted(native),required=sorted(required),admitted=[x.name for x in scene.agents])))
    EnvUnifiedSimulation._disable_offroad_agents=admission
    original=manifest.SelectionBinding
    class Binding(original):
        def __init__(self,*a,**kw):
            kw['history_num_frames']=0
            super().__init__(*a,**kw)
    manifest.SelectionBinding=Binding

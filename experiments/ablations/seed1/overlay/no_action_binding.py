"""Keep every benchmark scene runnable when no legal adversary exists."""
import json
import ccdiff.evaluation.selection_manifest as original

BaseBinding = original.SelectionBinding


class BenchmarkSelectionBinding(BaseBinding):
    def __init__(self, *args, **kwargs):
        record = args[1] if len(args) > 1 else kwargs['record']
        if record.get('no_action'):
            kwargs['history_num_frames'] = 0
        super().__init__(*args, **kwargs)

    def bind(self):
        if not self.record.get('no_action'):
            return super().bind()
        if self.record['adversary_vehicle_id'] != 'NO_ADVERSARY':
            raise ValueError('No-action sentinel must not pretend to be a vehicle token')
        names = list(self.env.current_agent_names)
        if names.count(self.record.get('ego_vehicle_id', 'ego')) != 1:
            raise ValueError('No unique original ego in no-action scene')
        if self._last_names is not None and names != self._last_names:
            raise ValueError('Agent order changed')
        self.ego_index = names.index(self.record.get('ego_vehicle_id', 'ego'))
        self.target_index = None
        self.control_policy.set_controllable_set([])
        if self._last_names is None:
            audit = dict(self.record, runtime_scene_index=self.scene_index,
                         runtime_target_index=None, runtime_ego_index=self.ego_index,
                         runtime_agent_names=names, dt_seconds=.1,
                         selection_bound=True, adversary_exists=False,
                         closed_loop_validated=False)
            self.audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
        self._last_names = names


original.SelectionBinding = BenchmarkSelectionBinding

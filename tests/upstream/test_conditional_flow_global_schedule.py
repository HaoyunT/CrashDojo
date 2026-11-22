import unittest
from dataclasses import dataclass

from torch import nn

from ccdiff.algos.algos import (
    CCDiffTrafficModel,
    _conditional_flow_runtime_objective_config,
)


class _ScheduledPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def set_conditional_flow_source_control(self, source_config, objective_config):
        self.calls.append(("set", source_config, objective_config))

    def clear_conditional_flow_source_control(self):
        self.calls.append(("clear",))


@dataclass(frozen=True)
class _Objective:
    current_global_step: int = 0


class ConditionalFlowGlobalScheduleTest(unittest.TestCase):
    def _model(self):
        model = CCDiffTrafficModel.__new__(CCDiffTrafficModel)
        nn.Module.__init__(model)
        model.nets = nn.ModuleDict({"policy": _ScheduledPolicy()})
        model.use_ema = False
        return model

    def test_schedule_is_identity_before_global_start(self):
        model = self._model()
        model.set_conditional_flow_source_control_schedule(
            "source", "objective", start_step=40, end_step=60
        )
        schedule = model._conditional_flow_source_control_schedule
        self.assertEqual(schedule["start_step"], 40)
        self.assertEqual(schedule["end_step"], 60)
        self.assertEqual(model.nets["policy"].calls, [("clear",)])

    def test_invalid_schedule_is_rejected(self):
        model = self._model()
        with self.assertRaises(ValueError):
            model.set_conditional_flow_source_control_schedule(
                "source", "objective", start_step=20, end_step=20
            )

    def test_runtime_objective_receives_simulator_global_step(self):
        model = self._model()
        objective = _Objective()
        model.set_conditional_flow_source_control_schedule(
            "source", objective, start_step=20, end_step=50
        )
        schedule = model._conditional_flow_source_control_schedule
        runtime = _conditional_flow_runtime_objective_config(
            schedule["objective_config"], 35
        )
        self.assertEqual(runtime.current_global_step, 35)
        self.assertEqual(objective.current_global_step, 0)


if __name__ == "__main__":
    unittest.main()

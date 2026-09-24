"""A NAVSIM agent that replays trajectories computed offline (by the zero-shot runners in their own venvs), so the
official devkit scores them unchanged. Works with navsim v1.1 and v2 (they differ only in AbstractAgent.__init__).

    run_pdm_score.py ... agent._target_=jevdrive.navsim_agent.PrecomputedAgent +agent.predictions=<file.npz>
The npz holds `tokens` (N,) and `poses` (N, 8, 3): x, y, yaw at 0.5 ... 4.0 s in the current rear-axle frame.
"""
import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory


class PrecomputedAgent(AbstractAgent):
    requires_scene = True   # the scene carries the token

    def __init__(self, predictions: str, trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)):
        torch.nn.Module.__init__(self)
        self.requires_scene = True
        self._trajectory_sampling = trajectory_sampling
        self._path = predictions
        self._poses = None

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        z = np.load(self._path)
        self._poses = dict(zip(z["tokens"].tolist(), z["poses"].astype(np.float32)))

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene=None) -> Trajectory:
        return Trajectory(self._poses[scene.scene_metadata.initial_token], self._trajectory_sampling)

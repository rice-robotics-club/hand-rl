import genesis as gs
import torch
from tensordict import TensorDict

from src.interfaces.vec_sim import VecSim


class RukaSim(VecSim):
    def __init__(self, num_envs: int):
        self._num_envs = num_envs
        self._dt = 1.0 / 60.0

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self._dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # For this locomotion policy, there are usually no more than 20 collision pairs. Setting a low value
                # can save memory. Violating this condition will raise an exception.
                max_collision_pairs=20,
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=True,
        )

        self._scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        self.robot = self._scene.add_entity(
            gs.morphs.URDF(
                file="robots/ruka_description/urdf/robot.urdf",
                pos=[0, 0, 0],
                quat=[0, 0, 0, 1],
            ),
        )

        self._scene.build(self._num_envs)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def state(self) -> TensorDict:
        return TensorDict()

    def step(self, actions: torch.Tensor):
        self._scene.step()

    def reset(self, idx: int | torch.Tensor | None = None):
        self._scene.reset(envs_idx=idx)

# Created by Xingyu Lin, 2019/1/8                                                                                  
from dfm_env.sawyer_float_env import SawyerFloatEnv
import numpy as np
from .utils.util import get_name_arr_and_len, ignored_physics_warning, cv_render


class RopeFloatEnv(SawyerFloatEnv):
    def __init__(self, distance_threshold=5e-2, **kwargs):
        model_path = 'tasks/rope_float.xml'
        super().__init__(model_path=model_path, distance_threshold=distance_threshold, **kwargs)

    # Rope specific helper functions
    def _sample_rope_init_pos(self):
        # Sample one of the rope elements and put the gripper on top of it
        sampled_idx = np.random.choice(self.rope_xpos_inds, 1)
        return self.physics.data.xpos[sampled_idx] + np.array([0, 0, 0.2])

    def _reset_sim(self):
        # Sample goal and render image, Get the goal after the environment is stable
        with self.physics.reset_context():
            self._random_push_rope()
            if self.use_image_goal or True:
                self.physics.data.qpos[self.state_target_rope_ref_inds[1]] += self.visualization_offset
                target_original_transparancy = self.physics.model.geom_rgba[self.target_rope_geom_rgba_inds, 3][0]
                self.physics.model.geom_rgba[self.target_rope_geom_rgba_inds, 3] = 1.
                self.physics.model.geom_rgba[self.rope_geom_rgba_inds, 3] = 0

                # self.physics.model.geom_rgba[1, :] = np.asarray([0., 0., 0, 0.])  # Make the goal transparent
                self.goal_observation = self.render(depth=False)
                self.physics.data.qpos[self.state_target_rope_ref_inds[1]] -= self.visualization_offset
                self.physics.model.geom_rgba[self.target_rope_geom_rgba_inds, 3] = target_original_transparancy
                self.physics.model.geom_rgba[self.rope_geom_rgba_inds, 3] = 1.
                # Set the target qpos
                # self.physics.data.qpos[self.state_target_rope_inds] = self.physics.data.qpos[self.state_rope_inds]
                # self.physics.data.qvel[self.state_target_rope_inds] = 0
        self.goal_state = self.get_target_goal_state()
        return True

    def _sample_rope_neighbourhood(self):
        # Sample one of the rope elements and put the gripper on top of it
        sampled_idx = np.random.choice(self.rope_xpos_inds, 1)
        point_start = self.physics.data.xpos[sampled_idx][0].copy()
        point_end = point_start.copy()
        delta_x = (np.random.random() - 0.5) / 10
        delta_y = np.sign(np.random.random() - 0.5) * 0.03
        delta_xy = [delta_x, delta_y]
        point_start[:2] += delta_xy
        point_end[:2] -= delta_xy
        # Only need this if the sampled neighbourhood is for the sawyer gripper
        # point_start[2] = self.boundary_range[2][0]  # Account for the height of the gripper
        # point_end[2] = self.boundary_range[2][0]  # Account for the height of the gripper
        return point_start, point_end

    def _random_push_rope(self):
        # Sample two points near the rope and let the block to move from one point to another point
        # Keep trying until the position of the rope changes
        init_rope_state = self.get_achieved_goal_state()
        while True:
            start_pt, end_pt = self._sample_rope_neighbourhood()
            self._move_arm_by_endpoints(start_pt, end_pt, render=False)
            cur_rope_state = self.get_achieved_goal_state()
            distance = np.linalg.norm(init_rope_state - cur_rope_state)
            if 1e-5 < distance < 3:
                # Set the target rope qpos and reset the rope qpos
                # self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)
                target_rope_qpos = self.physics.data.qpos[self.state_rope_inds].copy()
                self.physics.data.qpos[self.init_indexes] = self.init_qpos
                self.physics.data.qvel[self.init_indexes] = self.init_qvel
                self.physics.data.qpos[self.state_target_rope_inds] = target_rope_qpos
                self.set_arm_location(self.arm_init_pos)
                return
            else:
                # Restore the initial rope state
                self.physics.data.qpos[self.init_indexes] = self.init_qpos
                self.physics.data.qvel[self.init_indexes] = self.init_qvel
            break
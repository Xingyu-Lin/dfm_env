# Created by Xingyu Lin, 2019/1/8                                                                                  
from dfm_env.sawyer_float_env import SawyerFloatEnv
import numpy as np
from .utils.util import get_name_arr_and_len, ignored_physics_warning, cv_render
import os.path


class RopeFloatEnv(SawyerFloatEnv):
    def __init__(self, distance_threshold=5e-2, goal_push_num=2, visualization_mode=False, **kwargs):
        model_path = 'tasks/rope_float.xml'
        self.goal_push_num = goal_push_num
        cached_file = './dfm_env/data/generated_rope_{}.npz'.format(goal_push_num)
        if os.path.exists(cached_file):
            self.use_cached_inits_goals = True
            data = np.load(cached_file)
            self.all_init_qpos = data['all_init_qpos']
            self.all_target_qpos = data['all_target_qpos']
            print('Rope_float_env: using cached init poses and goal poses')
        else:
            self.use_cached_inits_goals = False
        self.visualization_mode = visualization_mode
        super().__init__(model_path=model_path, distance_threshold=distance_threshold, **kwargs)
        if not self.visualization_mode:
            # Hide the arm in case we are not visualizing
            self.physics.model.geom_rgba[self.geom_rgba_arm_inds, 3] = 0
            # Also hide the target rope if image observation is used
            if self.use_visual_observation:
                self.physics.model.geom_rgba[self.geom_rgba_target_rope_inds, 3] = 0.

    # Rope specific helper functions
    def _sample_rope_init_xpos(self):
        # Sample one of the rope elements and put the gripper on top of it
        sampled_idx = np.random.choice(self.xpos_rope_inds, 1)
        return self.physics.data.xpos[sampled_idx] + np.array([0, 0, 0.20])

    def _reset_sim(self):
        # Sample goal and render image, Get the goal after the environment is stable
        # print('camera:', self.get_camera_info(-1))

        with self.physics.reset_context():
            self._reset_all_to_init_pos()
            if not self.use_cached_inits_goals:
                self._random_push_rope(push_num=1)  # Init rope distribution
                one_push_rope_qpos = self.physics.data.qpos[self.qpos_rope_inds]
                # Set the target rope qpos and reset the rope qpos
                self._random_push_rope(push_num=None)  # Push k times according to the environment
                target_rope_qpos = self.physics.data.qpos[self.qpos_rope_inds].copy()
            else:
                cached_idx = np.random.randint(0, len(self.all_init_qpos))
                one_push_rope_qpos = self.all_init_qpos[cached_idx]
                target_rope_qpos = self.all_target_qpos[cached_idx]

            self.physics.data.qpos[self.qpos_rope_inds] = one_push_rope_qpos
            self.physics.data.qvel[self.qpos_rope_inds] = np.zeros(len(self.qpos_rope_inds))
            self.physics.data.qpos[self.qpos_target_rope_inds] = target_rope_qpos
            self.physics.forward()
            # self.set_arm_location(self.init_arm_xpos)

            if self.use_image_goal or True:
                # self.physics.data.qpos[self.qpos_target_rope_ref_inds[1]] += self.visualization_offset
                target_original_transparancy = self.physics.model.geom_rgba[self.geom_rgba_target_rope_inds, 3][0]
                self.physics.model.geom_rgba[self.geom_rgba_target_rope_inds, 3] = 1.
                self.physics.model.geom_rgba[self.geom_rgba_rope_inds, 3] = 0
                self.physics.forward()
                # self.physics.model.geom_rgba[1, :] = np.asarray([0., 0., 0, 0.])  # Make the goal transparent
                self.goal_observation = self.render(depth=False)
                # self.physics.data.qpos[self.qpos_target_rope_ref_inds[1]] -= self.visualization_offset
                self.physics.model.geom_rgba[self.geom_rgba_target_rope_inds, 3] = target_original_transparancy
                self.physics.model.geom_rgba[self.geom_rgba_rope_inds, 3] = 1.
                # Set the target qpos
                # self.physics.data.qpos[self.state_target_rope_inds] = self.physics.data.qpos[self.state_rope_inds]
                # self.physics.data.qvel[self.state_target_rope_inds] = 0
        self.goal_state = self.get_target_goal_state()
        return True

    def _sample_rope_neighbourhood(self):
        # Sample one of the rope elements and put the gripper on top of it
        sampled_idx = np.random.choice(self.xpos_rope_inds, 1)
        point_start = self.physics.data.xpos[sampled_idx][0].copy()
        point_end = point_start.copy()
        delta_x = (np.random.random() - 0.5) / 5
        delta_y = np.sign(np.random.random() - 0.5) * 0.2
        delta_xy = [delta_x, delta_y]
        point_start[:2] += delta_xy
        point_end[:2] -= delta_xy
        # Only need this if the sampled neighbourhood is for the sawyer gripper
        # point_start[2] = self.boundary_range[2][0]  # Account for the height of the gripper
        # point_end[2] = self.boundary_range[2][0]  # Account for the height of the gripper
        return point_start, point_end

    def _random_push_rope(self, push_num=None):
        # Sample two points near the rope and let the block to move from one point to another point
        # Keep trying until the position of the rope changes

        # init_rope_state = self.get_achieved_goal_state()
        if push_num is None:
            push_num = np.random.randint(0, self.goal_push_num) + 1

        for _ in range(push_num):
            start_pt, end_pt = self._sample_rope_neighbourhood()
            # Fix the height of the arm when moving
            start_pt[2] = end_pt[2] = self.arm_height
            self._move_arm_by_endpoints(start_pt, end_pt, render=False)
        return

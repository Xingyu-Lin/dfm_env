# Created by Xingyu Lin, 2018/12/6
import gym
import numpy as np

from .base import Base
from .utils.util import get_name_arr_and_len, ignored_physics_warning, cv_render
from dm_control import mujoco
from dm_control.suite import base
import cv2 as cv


class SawyerFloatEnv(Base, gym.utils.EzPickle):
    def __init__(self, model_path, distance_threshold, distance_threshold_obs=0,
                 n_substeps=20, horizon=200, image_size=400, action_type='endpoints',
                 with_goal=False,
                 use_visual_observation=True,
                 use_image_goal=True,
                 use_true_reward=False,
                 arm_height=0.85,  # TODO Check this later
                 arm_move_velocity=0.5,
                 **kwargs):
        '''

        :param model_path:
        :param distance_threshold:
        :param distance_threshold_obs:
        :param n_substeps:
        :param horizon:
        :param image_size:
        :param action_type:
            'endpoints': The agent will specify the starting and the ending point of a push (Both in 2D)
        :param with_goal:
        :param use_visual_observation:
        :param use_image_goal:
        :param use_true_reward:
        :param use_dof: ['both', 'arm', 'gripper']
        Base class for sawyer manipulation environments
        '''
        self.action_type = action_type
        if action_type == 'endpoints':
            n_actions = 4
            self.arm_height = arm_height
            self.arm_move_velocity = arm_move_velocity
        elif action_type == 'velocity':
            n_actions = 3
        else:
            assert NotImplementedError
        Base.__init__(self, model_path=model_path, n_substeps=n_substeps, horizon=horizon, n_actions=n_actions,
                      image_size=image_size, use_image_goal=use_image_goal,
                      use_visual_observation=use_visual_observation,
                      with_goal=with_goal, reward_type='sparse', distance_threshold=distance_threshold,
                      distance_threshold_obs=distance_threshold_obs, use_true_reward=use_true_reward, **kwargs)

        gym.utils.EzPickle.__init__(self)

    # TODO set the arm to be invisible
    # Implementation of functions from GoalEnvExt
    # ----------------------------
    def _set_action(self, ctrl):
        # self.physics.data.qvel[:] = np.zeros(shape=len(self.physics.data.qvel[:]))
        # return
        assert len(ctrl) == self.n_actions
        if self.action_type == 'velocity':
            ctrl /= self.n_substeps
            self._set_arm_velocity(ctrl)
        elif self.action_type == 'mocap':
            point_st = self._apply_endpoint_boundary(ctrl[:2])
            point_en = self._apply_endpoint_boundary(ctrl[3:])
            point_st.append()
            self._move_arm_by_endpoints(point_st, point_en)

    def _get_obs(self):
        if self.use_visual_observation:
            obs = self.render(depth=False)
        else:
            # thetas = self.physics.data.qpos[self.state_rope_rot_inds]
            obs = np.concatenate((self.physics.data.qpos[self.state_rope_ref_inds].copy(),
                                  self.physics.data.qpos[self.state_rope_rot_inds].copy()), axis=0)

        if self.use_image_goal:
            assert False
            desired_goal = self.goal_observation
            achieved_goal = obs
        else:
            desired_goal = self.goal_state
            achieved_goal = self.get_achieved_goal_state()

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy()
        }

    # noinspection PyAttributeOutsideInit
    def configure_indexes(self):
        # Arm and gripper action joints
        list_joints = get_name_arr_and_len(self.physics.named.data.ctrl, 0)[0]
        self.action_arm_inds = [idx for idx, s in enumerate(list_joints) if s.startswith('tj')]
        # No gripper for now; modify rope_float.xml later
        self.action_gripper_inds = [idx for idx, s in enumerate(list_joints) if 'gripper_joint_' in s]

        # qpos index for arm, gripper and rope
        list_qpos = get_name_arr_and_len(self.physics.named.data.qpos, 0)[0]
        self.state_arm_inds = [idx for idx, s in enumerate(list_qpos) if 'arm_slide' in s]
        self.state_gripper_inds = [idx for idx, s in enumerate(list_qpos) if 'gripper' in s]
        self.state_rope_ref_inds = [idx for idx, s in enumerate(list_qpos) if s == "Rope_ref"]
        self.state_rope_rot_inds = [idx for idx, s in enumerate(list_qpos) if s.startswith('Rope_J')]
        self.state_rope_inds = self.state_rope_ref_inds + self.state_rope_rot_inds
        self.state_target_rope_ref_inds = [idx for idx, s in enumerate(list_qpos) if s == "targetRope_ref"]
        self.state_target_rope_rot_inds = [idx for idx, s in enumerate(list_qpos) if s.startswith('targetRope_J')]
        self.state_target_rope_inds = self.state_target_rope_ref_inds + self.state_target_rope_rot_inds
        # self.state_push_block_inds = [idx for idx, s in enumerate(list_qpos) if 'push_block_slide' in s]
        # self.state_push_block_inds = self.state_push_block_inds[:2]

        list_ctrl = get_name_arr_and_len(self.physics.named.data.ctrl, 0)[0]
        self.ctrl_arm_inds = [idx for idx, s in enumerate(list_ctrl) if 'tj' in s]
        self.ctrl_gripper_inds = [idx for idx, s in enumerate(list_ctrl) if 'tg' in s]
        self.ctrl_rope_inds = [idx for idx, s in enumerate(list_ctrl) if 'tr' in s]

        list_geom = get_name_arr_and_len(self.physics.named.model.geom_rgba, 0)[0]
        self.rope_geom_rgba_inds = [idx for idx, s in enumerate(list_geom) if s != 0 and s.startswith('Rope_G')]
        self.target_rope_geom_rgba_inds = [idx for idx, s in enumerate(list_geom) if
                                           s != 0 and s.startswith('targetRope_G')]
        # self.push_block_geom_rgba_inds = [idx for idx, s in enumerate(list_geom) if
        #                                   s != 0 and s.startswith('pushBlock_G')]

        list_xpos = get_name_arr_and_len(self.physics.named.data.xpos, 0)[0]
        self.rope_xpos_inds = [idx for idx, s in enumerate(list_xpos) if s.startswith('Rope_')]
        self.target_rope_xpos_inds = [idx for idx, s in enumerate(list_xpos) if s.startswith('targetRope_')]
        self.visualization_offset = 0.1

    def _init_configure(self):
        self.boundary_range = [[-0.61, 0.5], [0.12, 0.85]]  # [min_val, max_val] for each of the dimension
        self.configure_indexes()
        n1 = len(self.state_arm_inds)
        n2 = len(self.state_gripper_inds)
        n3 = len(self.state_rope_rot_inds)
        init_state_rope_ref = [-0.15, 0.5, 0.705, 1, 0, 0, 0]

        self.arm_init_pos = self._sample_rope_init_pos()

        # self.arm_init_quat = [0, 1, 0, 0]
        self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))

        self.set_arm_location(self.arm_init_pos)
        init_arm_qpos = self.physics.data.qpos[self.state_arm_inds]
        self.init_indexes = self.state_arm_inds + self.state_rope_ref_inds + self.state_rope_rot_inds
        self.init_qpos = np.hstack([init_arm_qpos, init_state_rope_ref, np.zeros(n3)])
        self.init_qvel = np.zeros(len(self.init_indexes), )

    def get_current_info(self):
        """
        :return: The true current state, 'ag_state', and goal state, 'g_state'
        """
        info = {
            'ag_state': self.get_achieved_goal_state(),
            'g_state': self.goal_state.copy()
        }
        return info

    # Floating Sawyer environemtn specific helper functions
    # --------------------------------------------------
    def _apply_endpoint_boundary(self, point):
        for i in range(2):
            point[i] = np.clip(point[i], self.boundary_range[i][0], self.boundary_range[i][1])
        return point

    def get_arm_location(self):
        return self.physics.data.qpos[self.state_arm_inds]

    def set_arm_location(self, arm_location):
        self.physics.data.qpos[self.state_arm_inds] = arm_location

    def _set_arm_velocity(self, arm_velocity):
        self.physics.data.qvel[self.state_arm_inds] = arm_velocity

    def _move_arm_by_endpoints(self, arm_st_loc, arm_en_loc, velocity=None, render=False):
        # Move the arm from the start loc to the end loc
        if velocity is None:
            velocity = self.arm_move_velocity
        self.set_arm_location(arm_st_loc)
        move_dir = (arm_en_loc - arm_st_loc) * np.linalg.norm((arm_en_loc - arm_st_loc))
        self._set_arm_velocity(move_dir * velocity)
        prev_dist = 10000
        while True:
            if render:
                img = self.physics.render(camera_id='static_camera')
                cv_render(img)
            with ignored_physics_warning():
                self.physics.step()
            cur_dist = self._get_endpoints_distance(self.get_arm_location(), arm_en_loc)
            if cur_dist > prev_dist:
                break
            prev_dist = cur_dist

    def get_target_goal_state(self):
        # thetas = self.physics.data.qpos[self.state_target_rope_inds[7:]]
        # return np.hstack([np.cos(thetas), np.sin(thetas)])
        return self.physics.data.xpos[self.target_rope_xpos_inds, :2].flatten().copy()

    def get_achieved_goal_state(self):
        # ref_pose = self.physics.data.qpos[self.state_rope_ref_inds[-4:]]
        # thetas = self.physics.data.qpos[self.state_rope_inds[7:]]
        # return np.hstack([ref_pose, np.cos(thetas), np.sin(thetas)])
        # Only need to achieve the angles
        # return np.hstack([np.cos(thetas), np.sin(thetas)])
        return self.physics.data.xpos[self.rope_xpos_inds, :2].flatten().copy()

    @staticmethod
    def _get_endpoints_distance(pt1, pt2):
        return np.linalg.norm(pt1 - pt2)

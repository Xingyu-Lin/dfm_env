# Created by Xingyu Lin, 2018/12/6
import gym
import numpy as np

from .base import Base
from .utils.util import get_name_arr_and_len
from dm_control import mujoco
from dm_control.suite import base
import random


class SawyerEnv(Base, gym.utils.EzPickle):
    def __init__(self, model_path, distance_threshold, distance_threshold_obs=0,
                 n_substeps=20, horizon=200, image_size=400, action_type='mocap',
                 with_goal=False,
                 use_visual_observation=True,
                 use_image_goal=True,
                 use_true_reward=False, use_dof='both', fix_gripper=True, **kwargs):
        '''

        :param model_path:
        :param distance_threshold:
        :param distance_threshold_obs:
        :param n_substeps:
        :param horizon:
        :param image_size:
        :param action_type:
        :param with_goal:
        :param use_visual_observation:
        :param use_image_goal:
        :param use_true_reward:
        :param use_dof: ['both', 'arm', 'gripper']
        Base class for sawyer manipulation environments
        '''
        # TODO change n_action to be dependent on action_type

        self.use_dof = use_dof
        self.action_type = action_type
        self.fix_gripper = fix_gripper
        if action_type == 'mocap':
            n_actions = 4
            if fix_gripper:
                n_actions -= 1
        Base.__init__(self, model_path=model_path, n_substeps=n_substeps, horizon=horizon, n_actions=n_actions,
                      image_size=image_size, use_image_goal=use_image_goal,
                      use_visual_observation=use_visual_observation,
                      with_goal=with_goal, reward_type='sparse', distance_threshold=distance_threshold,
                      distance_threshold_obs=distance_threshold_obs, use_true_reward=use_true_reward, **kwargs)

        gym.utils.EzPickle.__init__(self)

    def get_current_info(self):
        """
        :return: The true current state, 'ag_state', and goal state, 'g_state'
        """
        info = {
            'ag_state': self.get_achieved_goal_state(),
            'g_state': self.goal_state.copy()
        }
        return info

    def reset_mocap2body_xpos(self):
        """Resets the position and orientation of the mocap bodies to the same
        values as the bodies they're welded to.
        """

        if (self.physics.model.eq_type is None or
                    self.physics.model.eq_obj1id is None or
                    self.physics.model.eq_obj2id is None):
            return

        for eq_type, obj1_id, obj2_id in zip(self.physics.model.eq_type,
                                             self.physics.model.eq_obj1id,
                                             self.physics.model.eq_obj2id):

            # if eq_type != mujoco_py.const.EQ_WELD:
            #    continue

            mocap_id = self.physics.model.body_mocapid[obj1_id]
            if mocap_id != -1:
                # obj1 is the mocap, obj2 is the welded body
                body_idx = obj2_id
            else:
                # obj2 is the mocap, obj1 is the welded body
                mocap_id = self.physics.model.body_mocapid[obj2_id]
                body_idx = obj1_id
            if mocap_id == -1:
                continue

            self.physics.data.mocap_pos[mocap_id][:] = self.physics.data.xpos[body_idx]
            self.physics.data.mocap_quat[mocap_id][:] = self.physics.data.xquat[body_idx]

    def _apply_mocap_boundary(self, ctrl):
        for i in range(3):
            if self.physics.data.mocap_pos[0][i] + ctrl[i] < self.boundary_range[i][0] and ctrl[i] < 0:
                ctrl[i] = 0
            elif self.physics.data.mocap_pos[0][i] + ctrl[i] > self.boundary_range[i][1] and ctrl[i] > 0:
                ctrl[i] = 0
        return ctrl

    def _set_action(self, ctrl):
        # todo ADD argument for mocap constrained 2d
        ctrl /= self.n_substeps
        # ctrl *=0
        # print('mocap:' , self.physics.data.mocap_pos[0], self.physics.data.mocap_quat[0])
        # print('gripper', self.physics.data.xpos[26], self.physics.data.xquat[26])
        if self.action_type == 'torque':
            assert False
            if self.use_dof == 'both':
                self.physics.data.ctrl[:] = ctrl
            elif self.use_dof == 'arm':
                self.physics.data.ctrl[self.action_arm_inds] = ctrl
            else:
                self.physics.data.ctrl[self.action_gripper_inds] = ctrl + 1
        elif self.action_type == 'velocity':
            assert False
            self.physics.data.ctrl[:] = 0
            if self.use_dof == 'both':
                self.physics.data.qvel[0:len(ctrl)] = ctrl
            elif self.use_dof == 'arm':
                self.physics.data.qvel[self.action_arm_inds] = ctrl
            else:
                self.physics.data.qvel[self.action_gripper_inds] = ctrl
        elif self.action_type == 'mocap':
            ctrl = self._apply_mocap_boundary(ctrl)
            # print('mocap pos:', self.physics.data.mocap_pos[0][:])
            # ctrl[2] = 0
            # if self.time_step % 50 == 0:
            #     self.reset_mocap2body_xpos()
            self.physics.data.mocap_quat[:] = self.gripper_init_quat
            self.physics.data.mocap_pos[0, :3] += ctrl[:3]
            if not self.fix_gripper:
                self.physics.data.ctrl[self.ctrl_gripper_indices] = ctrl[3:]
                # else:
                #     self.physics.named.data.qpos[self.state_gripper_inds] = 0
                # print(self._distance_between_gripper_rope_ref())
                # if self._distance_between_gripper_rope_ref() > 0.4 :
                #     self.gripper_init_pos = self._sample_rope_init_pos()
                #     self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)

    def _move_gripper(self, gripper_target, gripper_rotation=None):
        self.physics.data.mocap_pos[0][:] = gripper_target
        if gripper_rotation is not None:
            self.physics.data.mocap_quat[0][:] = gripper_rotation
        prev_dist = 10000
        while True:
            cur_dist = self._get_gripper_mocap_distance(gripper_target)
            self.physics.step()
            if np.abs(cur_dist - prev_dist) < 1e-5:
                return
            prev_dist = cur_dist

    def _get_gripper_mocap_distance(self, gripper_target):
        gripper_base_xpos = self.physics.named.data.xpos['arm_gripper_base']
        return np.linalg.norm(gripper_base_xpos - gripper_target)

import collections

from dm_control import mujoco, viewer
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np
import cv2
from os import path
import math
import os
from gym import Env, GoalEnv
from dm_control.mujoco import Physics
from gym import spaces
from dm_control.utils import io as resources


class RopeEnv(GoalEnv):
    def __init__(self, model_path='rope.xml', distance_threshold=5e-2, frame_skip=2,
                 horizon=100, goal_range=None, image_size=512, action_type='torque', use_visual_observation=True,
                 camera_name='static_camera', use_dof='both'):
        '''

        :param model_path:
        :param distance_threshold:
        :param frame_skip:
        :param horizon:
        :param goal_range:
        :param image_size:
        :param action_type:
        :param use_visual_observation:
        :param camera_name:
        :param use_dof: ['both', 'arm', 'gripper']
        Base class for sawyer manipulation environments
        '''

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self.physics = Physics.from_xml_string(*self.get_model_and_assets(fullpath))
        self.np_random = None
        self.camera_name = camera_name
        self.data = self.physics.data
        self.viewer = None
        self.distance_threshold = distance_threshold
        self.frame_skip = frame_skip
        self.reward_type = 'sparse'
        self.horizon = horizon
        self.use_visual_observation = use_visual_observation
        self._max_episode_steps = horizon
        self.use_dof = use_dof
        self.time_step = 0
        self.image_size = image_size
        self.action_type = action_type
        self.goal_range = goal_range if goal_range is not None else [-0.16, 0.16]
        self.configure_indexes()

        if self.use_dof == 'both':
            self.action_length = mujoco.action_spec(self.physics).shape[0]
        elif self.use_dof == 'arm':
            self.action_length = len(self.arm_inds)
        elif self.use_dof == 'gripper':
            self.action_length = len(self.gripper_inds)
        else:
            assert 'Wrong arguments'

        obs = self.reset()
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(self.action_length,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.goal_dim = np.prod(obs['achieved_goal'].shape)

    def configure_indexes(self):

        list_joints = self.physics.named.data.ctrl.axes.row.names
        self.arm_inds = [idx for idx, s in enumerate(list_joints) if 'tj' in s]
        self.gripper_inds = [idx for idx, s in enumerate(list_joints) if 'tg' in s]


    def reset(self):
        self.time_step = 0
        # with self.physics.reset_context()
        # TODO : reset gripper location here
        self.physics.forward()
        return self.get_current_observation()

    def get_current_observation(self):

        if self.use_visual_observation == 'image':
            obs = self.physics.render(height=self.image_size, width=self.image_size, camera_id=self.camera_name)
        else:
            obs = np.concatenate((self.physics.data.qpos.copy(), self.physics.data.qvel.copy()), axis=0)

        # TODO: Find a way to specify goal location
        desired_goal = self.get_fake_goal_location()
        achieved_goal = self.get_fake_goal_location()

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy()
        }

    def get_fake_goal_location(self):
        return np.zeros(5)

    def set_control(self, ctrl):

        assert len(ctrl) == self.action_length, "Action vector not of right length"
        if self.action_type == 'torque':
            if self.use_dof == 'both':
                self.physics.data.ctrl[:] = ctrl
            elif self.use_dof == 'arm':
                self.physics.data.ctrl[self.arm_inds] = ctrl
            else:
                self.physics.data.ctrl[self.gripper_inds] = ctrl + 1
        else:
            self.physics.data.ctrl[:] = 0
            if self.use_dof == 'both':
                self.physics.data.qvel[0:len(ctrl)] = ctrl
            elif self.use_dof == 'arm':
                self.physics.data.qvel[self.arm_inds] = ctrl
            else:
                self.physics.data.qvel[self.gripper_inds] = ctrl

    def step(self, ctrl):

        ctrl = np.clip(ctrl, -np.inf, np.inf)
        self.set_control(ctrl)
        for _ in range(self.frame_skip):
            self.physics.step()
        self.time_step += 1
        obs = self.get_current_observation()
        reward = 0.0
        done = False
        if self.time_step >= self.horizon:
            done = True
        info = {}
        return obs, reward, done, info

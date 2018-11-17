
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
                 horizon=100, goal_range=[-0.16, 0.16], image_size=400, action_type='torque', obs_type='image', camera_name = 'static_camera', use_dof = 'both'):

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
        self.obs_type = obs_type
        self._max_episode_steps = horizon
        self.use_dof = use_dof
        self.time_step = 0
        self.image_size = image_size
        self.action_type = action_type
        self.goal_range = goal_range
        self.configure_indexes()

        if self.use_dof == 'both':
            self.action_length = mujoco.action_spec(self.physics).shape[0]
        elif self.use_dof == 'arm':
            self.action_length = len(self.arm_inds)
        else:
            self.action_length = len(self.gripper_inds)

        obs = self.reset()
        self.action_space = spaces.Box(-np.inf, np.inf, shape= (self.action_length,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.goal_dim = np.prod(obs['achieved_goal'].shape)

    def read_model(self,model_filename):

        """Reads a model XML file and returns its contents as a string."""
        return resources.GetResource(model_filename)

    def get_model_and_assets(self,model_filename):

        """Returns a tuple containing the model XML string and a dict of assets."""
        return self.read_model(model_filename = model_filename), common.ASSETS

    def configure_indexes(self):

        list_joints = self.physics.named.data.ctrl.axes.row.names
        self.arm_inds = [idx for idx, s in enumerate(list_joints) if 'tj' in s]
        self.gripper_inds = [idx for idx, s in enumerate(list_joints) if 'tg' in s]

    def set_camera_location(self, camera_id=None, pos=[0.0, 0.0, 0.0]):
        self.physics.model.cam_pos[camera_id] = pos

    def set_camera_fov(self, camera_id=None, fovy=50.0):
        self.physics.model.cam_fovy[camera_id] = fovy

    def set_camera_orientation(self, camera_id=None, orientation_quat=[0., 0., 0., 0.]):
        self.physics.model.cam_quat[camera_id] = orientation_quat

    def reset(self):

        self.time_step = 0
        #with self.physics.reset_context()
        #TODO : reset gripper location here
        self.physics.forward()
        return self.get_current_observation()

    def get_current_observation(self):

        if self.obs_type == 'image':
            obs = self.physics.render(height=self.image_size, width=self.image_size, camera_id= self.camera_name)
        else:
            obs = np.concatenate((self.physics.data.qpos.copy(), self.physics.data.qvel.copy()), axis = 0)

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
                self.physics.data.ctrl[self.gripper_inds] = ctrl+1
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



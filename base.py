# Created by Xingyu Lin, 2018/11/18
import os
from os import path
import numpy as np

from gym import GoalEnv
from gym import error, spaces
from gym.utils import seeding

from dm_control import mujoco, viewer
from dm_control.rl import control
from dm_control.suite import base
from dm_control.rl.control import PhysicsError
from dm_control.suite import common
from dm_control.mujoco import Physics
from dm_control.utils import io as resources
from termcolor import colored
import pickle
import cv2 as cv
import copy
import tensorflow as tf

class Base(GoalEnv):
    '''
    Base class for all dm_control based, goal oriented environments.
    Suppport different auxiliary tasks
    '''

    def __init__(self, model_path, n_substeps, n_actions, horizon, image_size, use_image_goal,
                 use_visual_observation, with_goal,
                 reward_type, distance_threshold, distance_threshold_obs, use_true_reward,
                 default_camera_name='static_camera', use_auxiliary_loss=False, state_estimation_reward_noise=0.,
                 state_estimation_input_noise=0., estimator_path=None, **kwargs):

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "./assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.state_estimation_reward_noise = state_estimation_reward_noise
        self.state_estimation_input_noise = state_estimation_input_noise
        self.estimator_path = estimator_path
        if estimator_path is not None:
            self.load_estimator(estimator_path)
        self.physics = Physics.from_xml_string(*self.get_model_and_assets(fullpath))

        self.n_actions = n_actions
        self.action_space = spaces.Box(-1, 1, shape=(self.n_actions,), dtype='float32')
        self._init_configure()
        self.np_random = None
        self.seed()

        # self._env_setup(initial_qpos=initial_qpos)
        # TODO
        # self.initial_state = copy.deepcopy(self.sim.get_state())
        self.n_substeps = n_substeps
        self.horizon = horizon
        self.image_size = image_size
        self.use_image_goal = use_image_goal
        self.use_visual_observation = use_visual_observation
        self.with_goal = with_goal

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.distance_threshold_obs = distance_threshold_obs
        self.use_true_reward = use_true_reward

        self._max_episode_steps = horizon
        self.time_step = 0

        # TODO
        # self.init_qpos = self.sim.data.qpos.ravel().copy()
        # self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.goal_state = self.goal_observation = self.goal_observation_estimated_state = None

        if (not use_visual_observation and distance_threshold == 0. and not use_true_reward) or (
          use_visual_observation and distance_threshold_obs == 0. and not use_true_reward):
            self.compute_reward = self.compute_reward_zero

        self.default_camera_name = default_camera_name
        self._set_camera()

        # TODO add this as an argument
        self.use_auxiliary_loss = use_auxiliary_loss
        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.goal_dim = np.prod(obs['achieved_goal'].shape)
        self.goal_state_dim = np.prod(self.goal_state.shape)

    def load_estimator(self, estimator_file):
        with open(estimator_file, 'rb') as f:
            self.estimator = pickle.load(f)
        # exit()

    def estimate_state(self, obs_img=None, goal_img=None):
        if obs_img is None:
            return self.estimator.sess.run([self.estimator.main.q_x_g],
                                           feed_dict={self.estimator.main.g_tf: goal_img.reshape(1, -1)})[0]
        if goal_img is None:
            return self.estimator.sess.run([self.estimator.main.q_x_o],
                                           feed_dict={self.estimator.main.o_tf: obs_img.reshape(1, -1)})[0]
        if obs_img is not None and goal_img is not None:
            return self.estimator.sess.run([self.estimator.main.q_x_o, self.estimator.main.q_x_g],
                                           feed_dict={self.estimator.main.o_tf: obs_img.reshape(1, -1),
                                                      self.estimator.main.g_tf: goal_img.reshape(1, -1)})

    def read_model(self, model_filename):

        """Reads a model XML file and returns its contents as a string."""
        return resources.GetResource(model_filename)

    def get_model_and_assets(self, model_filename):

        """Returns a tuple containing the model XML string and a dict of assets."""
        return self.read_model(model_filename=model_filename), common.ASSETS

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def set_state(self, qpos, qvel):
    #     self.physics.set_state()
    #     self.physics.get_state()
    #     old_state = self.sim.get_state()
    #     new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
    #                                      old_state.act, old_state.udd_state)
    #     self.sim.set_state(new_state)
    #     self.sim.forward()

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal. (only one of them)
        if self.use_true_reward:
            if info is not None:
                achieved_goal = info['ag_state']
                desired_goal = info['g_state']
            achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
            desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
            d_threshold = self.distance_threshold
        else:
            achieved_goal = achieved_goal.reshape([-1, self.goal_dim])
            desired_goal = desired_goal.reshape([-1, self.goal_dim])
            d_threshold = self.distance_threshold_obs

        if self.state_estimation_reward_noise > 0:
            noise1 = self.state_estimation_reward_noise * np.random.randn(*achieved_goal.shape)  # gaussian noise
            noise2 = self.state_estimation_reward_noise * np.random.randn(*desired_goal.shape)  # gaussian noise
            achieved_goal = achieved_goal + noise1
            desired_goal = desired_goal + noise2

        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == 'sparse':
            return -(d > d_threshold).astype(np.float32)
        else:
            return -d

    def compute_reward_zero(self, achieved_goal, desired_goal, info):
        if self.use_true_reward:
            if info is not None:
                achieved_goal = info['ag_state']
                desired_goal = info['g_state']
            achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
            desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
        else:
            achieved_goal = achieved_goal.reshape([-1, self.goal_dim])
            desired_goal = desired_goal.reshape([-1, self.goal_dim])
        assert achieved_goal.shape == desired_goal.shape
        return np.alltrue(np.equal(achieved_goal, desired_goal), axis=-1) - 1.

    # methods to override:
    # ----------------------------
    def _init_configure(self):
        pass

    def _reset_sim(self, restore_info=None):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        raise NotImplementedError

    def _get_obs(self):
        """
        Get observation
        """
        raise NotImplementedError

    def _set_action(self, ctrl):
        """
        Do simulation
        """
        raise NotImplementedError

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _set_camera(self):
        pass

    def get_current_info(self):
        """
        :return: The true current state, 'ag_state', and goal state, 'g_state'
        """
        return {}

    def _viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    # def _env_setup(self, initial_qpos):
    #     """Initial configuration of the environment. Can be used to configure initial state
    #     and extract information from the simulation.
    #     """
    #     pass

    def set_hidden_goal(self):
        """
        Hide the goal position from image observation
        """
        pass

    def get_image_obs(self, depth=True, hide_overlay=True, camera_id=-1):
        assert False
        return

    def _sample_goal_state(self):
        """Samples a new goal in state space and returns it.
        """
        return None, None

    # Core functions framework
    # -----------------------------

    def reset(self, restore_info=None):
        '''
        Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.
        '''
        self.time_step = 0
        if not self.with_goal:
            self.set_hidden_goal()

        goal_state, _ = self._sample_goal_state()
        if goal_state is None:
            self.goal_state = None
        else:
            self.goal_state = goal_state.copy()
        did_reset_sim = False
        while not did_reset_sim:
            if hasattr(self, 'get_restore_info') and restore_info is not None:
                did_reset_sim = self._reset_sim(restore_info)
            else:
                did_reset_sim = self._reset_sim()

        return self._get_obs()

    def step(self, action):
        action = action.flatten()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if hasattr(self, 'visualization_mode') and self.visualization_mode:
            ret = self._set_action(action)
            if ret != 'env_no_step':
                try:
                    for _ in range(self.n_substeps):
                        self.physics.step()
                except PhysicsError as ex:
                    print(colored(ex, 'red'))
            self._step_callback()
            aug_info = {}
            if (hasattr(self, 'prev_action_finished') and self.prev_action_finished) or (
              not hasattr(self, 'prev_obs')):
                obs = self._get_obs()
                self.prev_obs = obs
            else:
                obs = self.prev_obs
        else:
            ret = self._set_action(action)
            if ret != 'env_no_step':
                try:
                    for _ in range(self.n_substeps):
                        self.physics.step()
                except PhysicsError as ex:
                    print(colored(ex, 'red'))
            self._step_callback()
            obs = self._get_obs()
            if self.use_auxiliary_loss:
                # transformed_img, transformation = self.random_image_transformation(next_frame)
                if hasattr(self, 'equivalent_action_taken') and self.equivalent_action_taken is not None:
                    action_taken = self.equivalent_action_taken
                else:
                    action_taken = action
                aug_info = {
                    'action_taken': action_taken,
                    # 'transformed_frame': transformed_img.flatten(),
                    # 'transformation': transformation
                }
            else:
                aug_info = {}
        state_info = self.get_current_info()
        info = {**aug_info, **state_info}

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        self.time_step += 1
        # Episode ends only when the horizon is reached
        done = False
        if self.time_step >= self.horizon:
            done = True

        noise1 = self.state_estimation_input_noise * np.random.randn(*obs['achieved_goal'].shape)  # gaussian noise
        noise2 = self.state_estimation_input_noise * np.random.randn(*obs['desired_goal'].shape)  # gaussian noise
        obs['achieved_goal'] = obs['achieved_goal'] + noise1
        obs['desired_goal'] = obs['desired_goal'] + noise2
        return obs, reward, done, info

    def get_initial_info(self):
        state_info = self.get_current_info()

        if self.use_auxiliary_loss:
            aug_info = {
                'action_taken': np.zeros(self.action_space.shape),
                # 'transformed_frame': transformed_img.flatten(),
                # 'transformation': transformation
            }
            return {**aug_info, **state_info}
        else:
            return state_info

    def render(self, image_size=None, depth=False, camera_name=None):
        self._render_callback()
        if camera_name is None:
            camera_name = self.default_camera_name
        if image_size is None:
            image_size = self.image_size
        return self.physics.render(height=image_size, width=image_size, camera_id=camera_name, depth=depth)

    # Auxiliary Reward Methods
    # ----------------------------
    @staticmethod
    def random_image_transformation(image, max_translation=10, max_angle=30):
        angle = np.random.uniform(-max_angle, max_angle)
        translation_x = np.random.uniform(-max_translation, max_translation)
        translation_y = np.random.uniform(-max_translation, max_translation)

        width = image.shape[1]
        height = image.shape[0]
        M1 = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

        M2 = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

        transformed_img = cv.warpAffine(image, M1 + M2, (image.shape[1], image.shape[0]))
        return transformed_img, np.asarray([angle, translation_x, translation_y])

    # Helper Functions
    # ----------------------------
    def _get_info_state(self, achieved_goal, desired_goal):
        # Given g, ag in state space and return the distance and success
        achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
        desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return d, (d <= self.distance_threshold).astype(np.float32)

    def _get_info_obs(self, achieved_goal_obs, desired_goal_obs):
        # Given g, ag in state space and return the distance and success
        achieved_goal_obs = achieved_goal_obs.reshape([-1, self.goal_dim])
        desired_goal_obs = desired_goal_obs.reshape([-1, self.goal_dim])
        d = np.linalg.norm(achieved_goal_obs - desired_goal_obs, axis=-1)
        return d, (d <= self.distance_threshold_obs).astype(np.float32)

    def set_camera_location(self, camera_id=None, pos=None):
        if pos is None:
            pos = [0.0, 0.0, 0.0]
        self.physics.model.cam_pos[camera_id] = pos

    def set_camera_fov(self, camera_id=None, fovy=50.0):
        self.physics.model.cam_fovy[camera_id] = fovy

    def set_camera_orientation(self, camera_id=None, orientation_quat=None):
        if orientation_quat is None:
            orientation_quat = [0., 0., 0., 0.]
        self.physics.model.cam_quat[camera_id] = orientation_quat

    def get_camera_info(self, camera_id):
        '''
        :return: the camera position and quaternion
        '''
        return self.physics.model.cam_pos[camera_id], self.physics.model.cam_quat[camera_id]

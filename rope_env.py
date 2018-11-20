# Created by Xingyu Lin, 2018/11/18                                                                                  
import gym
import numpy as np

from base import Base
from utils.util import get_name_arr_and_len
from dm_control import mujoco
from dm_control.suite import base

init_qpos = {
    'arm_j0': 0.,
    'arm_j1': 0.,
    'arm_j2': 0.,
    'arm_j3': 0.,
    'arm_j4': 0.,
    'arm_j5': 0.,
    'arm_j6': 0.,
    'gripper_jl': 0.,
    'gripper_jr': 0.,
}

init_qvel = {
    'arm_j0': 0.,
    'arm_j1': 0.,
    'arm_j2': 0.,
    'arm_j3': 0.,
    'arm_j4': 0.,
    'arm_j5': 0.,
    'arm_j6': 0.,
    'gripper_jl': 0.,
    'gripper_jr': 0.,
}


class RopeEnv(Base, gym.utils.EzPickle):
    def __init__(self, model_path='tasks/rope.xml', distance_threshold=1e-2, distance_threshold_obs=0, n_substeps=20,
                 n_actions=8, horizon=50, image_size=400, action_type='torque',
                 with_goal=False,
                 use_visual_observation=True,
                 use_image_goal=True,
                 use_true_reward=False, use_dof='both'):
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

        Base.__init__(self, model_path=model_path, n_substeps=n_substeps, horizon=horizon, n_actions=n_actions,
                      image_size=image_size, use_image_goal=use_image_goal,
                      use_visual_observation=use_visual_observation,
                      with_goal=with_goal, reward_type='sparse', distance_threshold=distance_threshold,
                      distance_threshold_obs=distance_threshold_obs, use_true_reward=use_true_reward)

        gym.utils.EzPickle.__init__(self)

    # Implementation of functions from GoalEnvExt
    # ----------------------------
    def _init_configure(self):
        self.configure_indexes()
        self.n_actions = 8

    def _reset_sim(self):
        # Sample goal and render image

        with self.physics.reset_context():
            self.physics.data.qpos[:] = np.zeros(len(self.physics.data.qpos))
            self.physics.data.qvel[:] = np.zeros(len(self.physics.data.qvel))
            self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))

        # qpos = self.np_random.uniform(low=-2 * np.pi, high=2 * np.pi, size=self.model.nq)
        # self.set_state(qpos, qvel=self.init_qvel)
        # self.goal_state = self.get_end_effector_location()
        #
        # qpos[-2:] = self.goal_state
        # qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        # qvel[-2:] = 0
        # self.set_state(qpos, qvel)
        # self.goal_observation = self.render(mode='rgb_array', depth=False)
        # qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # qpos[-2:] = self.goal_state
        # qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        # qvel[-2:] = 0
        # self.set_state(qpos, qvel)

        return True

    def _sample_goal_state(self):
        """Samples a new goal in state space and returns it.
        """
        return []

    def _get_obs(self):
        if self.use_visual_observation:
            obs = self.render(depth=False)
        else:
            obs = np.concatenate((self.physics.data.qpos.copy(), self.physics.data.qvel.copy()), axis=0)

        # TODO Figure out how to specify goal

        if self.use_image_goal:
            assert False
            desired_goal = self.goal_observation
            achieved_goal = obs
        else:
            desired_goal = self.goal_state
            achieved_goal = self.rope_state

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy()
        }

    # def get_current_info(self):
    #     """
    #     :return: The true current state, 'ag_state', and goal state, 'g_state'
    #     """
    #     info = {
    #         'ag_state': self.get_end_effector_location().copy(),
    #         'g_state': self.get_goal_location().copy()
    #     }
    #     return info

    def _set_action(self, ctrl):
        if self.action_type == 'torque':
            if self.use_dof == 'both':
                self.physics.data.ctrl[:] = ctrl
            elif self.use_dof == 'arm':
                self.physics.data.ctrl[self.action_arm_inds] = ctrl
            else:
                self.physics.data.ctrl[self.action_gripper_inds] = ctrl + 1
        else:
            self.physics.data.ctrl[:] = 0
            if self.use_dof == 'both':
                self.physics.data.qvel[0:len(ctrl)] = ctrl
            elif self.use_dof == 'arm':
                self.physics.data.qvel[self.action_arm_inds] = ctrl
            else:
                self.physics.data.qvel[self.action_gripper_inds] = ctrl

    def get_current_info(self):
        return {}

    # def set_hidden_goal(self):
    #     self.sim.model.geom_rgba[9, :] = np.asarray([0., 0., 0, 0.])  # Make the goal transparent

    # Env specific helper functions
    # ----------------------------
    @property
    def rope_state(self):
        return self.physics.data.qpos[self.state_rope_inds]

    def configure_indexes(self):
        # Arm and gripper action ind
        list_joints = get_name_arr_and_len(self.physics.named.data.ctrl, 0)[0]
        self.action_arm_inds = [idx for idx, s in enumerate(list_joints) if 'tj' in s]
        self.action_gripper_inds = [idx for idx, s in enumerate(list_joints) if 'tg' in s]

        # Rope
        list_qpos = get_name_arr_and_len(self.physics.named.data.qpos, 0)[0]

        self.state_rope_inds = [idx for idx, s in enumerate(list_qpos) if s[0] == "J"]

    # def action_spec(self):
    #     """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    #     return mujoco.action_spec(self.physics)
    #
    # def get_observation(self):
    #     pass
    #
    # def get_reward(self):
    #     pass
    # def initialize_episode(self, physics):
    #     pass

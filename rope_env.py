# Created by Xingyu Lin, 2018/11/18                                                                                  
import gym
import numpy as np

from base import Base
from dm_control import mujoco
from dm_control.suite import base

class RopeEnv(Base, gym.utils.EzPickle):
    def __init__(self, model_path='tasks/rope.xml', distance_threshold=1e-2, distance_threshold_obs=0, n_substeps=20,
                 horizon=50, image_size=400, action_type='torque',
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

        Base.__init__(self, model_path=model_path, n_substeps=n_substeps, horizon=horizon, image_size=image_size,
                      use_image_goal=use_image_goal, use_visual_observation=use_visual_observation,
                      with_goal=with_goal, reward_type='sparse', distance_threshold=distance_threshold,
                      distance_threshold_obs=distance_threshold_obs, use_true_reward=use_true_reward, n_actions=8)

        gym.utils.EzPickle.__init__(self)

        # self.configure_indexes()
    # Implementation of functions from GoalEnvExt
    # ----------------------------

    def _reset_sim(self):
        # Sample goal and render image

        # with self.physics.reset_context():
            # physics.data.qpos[:] =
            # physics.data.qvel[:] =
            # physics.data.ctrl[:] =

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

    def _get_obs(self):
        if self.use_visual_observation:
            obs = self.render(depth=False)
        else:
            # TODO
            obs = np.concatenate((self.physics.data.qpos.copy(), self.physics.data.qvel.copy()), axis=0)

        # TODO Figure out how to specify goal
        desired_goal = achieved_goal = np.zeros(5)
        # if self.use_image_goal:
        #     desired_goal = self.goal_observation
        #     achieved_goal = obs
        # else:
        #     desired_goal = self.get_goal_location()
        #     achieved_goal = self.get_end_effector_location()

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

    def get_current_info(self):
        return {}
    # def set_hidden_goal(self):
    #     self.sim.model.geom_rgba[9, :] = np.asarray([0., 0., 0, 0.])  # Make the goal transparent


    # Env specific helper functions
    # ----------------------------
    def configure_indexes(self):

        list_joints = self.physics.named.data.ctrl.axes.row.names
        self.arm_inds = [idx for idx, s in enumerate(list_joints) if 'tj' in s]
        self.gripper_inds = [idx for idx, s in enumerate(list_joints) if 'tg' in s]

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

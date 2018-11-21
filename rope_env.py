# Created by Xingyu Lin, 2018/11/18                                                                                  
import gym
import numpy as np

from .base import Base
from .utils.util import get_name_arr_and_len
from dm_control import mujoco
from dm_control.suite import base


class RopeEnv(Base, gym.utils.EzPickle):
    def __init__(self, model_path='tasks/rope.xml', distance_threshold=1e-2, distance_threshold_obs=0, n_substeps=5,
                 n_actions=2, horizon=50, image_size=400, action_type='mocap',
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
        #self.n_actions = len(self.physics.data.ctrl)
        n1 = len(self.state_arm_inds)
        n2 = len(self.state_gripper_inds)
        n3 = len(self.state_rope_rot_inds)
        init_state_rope_ref = [-0.15, 0, 0.92, 1, 0, 0, 0]
        self.init_qpos = np.hstack([np.zeros(n1 + n2), init_state_rope_ref, np.zeros(n3)])
        self.init_qvel = np.zeros(len(self.physics.data.qvel), )

    def _reset_sim(self):
        # Sample goal and render image

        with self.physics.reset_context():
            self.physics.data.qpos[:] = self.init_qpos
            self.physics.data.qvel[:] = self.init_qvel
            self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))
        self.physics.step()

        # Get the goal after the environment is stable
        self.goal_state, goal_theta = self._sample_goal_state()
        with self.physics.reset_context():
            self.physics.data.qpos[self.state_rope_rot_inds] = goal_theta
            self.physics.data.qpos[self.state_rope_ref_inds[2]] += 0.2
            for _ in range(1000):
                self.physics.step()
        self.goal_state = self.get_achieved_goal_state()

        with self.physics.reset_context():
            self.physics.data.qpos[:] = self.init_qpos
            self.physics.data.qvel[:] = self.init_qvel
            self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))

        if self.use_image_goal:
            self.goal_observation = self.render(depth=False)

        # qpos = self.np_random.uniform(low=-2 * np.pi, high=2 * np.pi, size=self.model.nq)
        # self.set_state(qpos, qvel=self.init_qvel)
        # self.goal_state = self.get_end_effector_location()
        #
        # qpos[-2:] = self.goal_state
        # qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        # qvel[-2:] = 0
        # self.set_state(qpos, qvel)
        #
        # qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # qpos[-2:] = self.goal_state
        # qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        # qvel[-2:] = 0
        # self.set_state(qpos, qvel)

        return True

    def _sample_goal_state(self):
        """Samples a new goal in state space and returns it.
        """
        n = len(self.state_rope_rot_inds)
        thetas = np.random.random(n, ) * 0.1
        # thetas = np.random.random(n, ) * 2 * np.pi

        goal_state = np.hstack([self.init_qpos[self.state_rope_ref_inds], np.cos(thetas), np.sin(thetas)])
        # TODO Do forward dynamics to get the achiavable goal state
        # self.set_state()
        return goal_state, thetas

    def _get_obs(self):
        if self.use_visual_observation:
            obs = self.render(depth=False)
        else:
            thetas = self.physics.data.qpos[self.state_rope_rot_inds]
            obs = np.concatenate((self.physics.data.qpos[self.state_arm_inds].copy(),
                                  self.physics.data.qpos[self.state_gripper_inds].copy(),
                                  self.physics.data.qpos[self.state_rope_ref_inds].copy(),
                                  np.cos(thetas), np.sin(thetas), self.physics.data.qvel.copy()), axis=0)

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

    # def get_current_info(self):
    #     """
    #     :return: The true current state, 'ag_state', and goal state, 'g_state'
    #     """
    #     info = {
    #         'ag_state': self.get_end_effector_location().copy(),
    #         'g_state': self.get_goal_location().copy()
    #     }
    #     return info
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

            #if eq_type != mujoco_py.const.EQ_WELD:
            #    continue

            mocap_id = self.physics.model.body_mocapid[obj1_id]
            if mocap_id != -1:
                # obj1 is the mocap, obj2 is the welded body
                body_idx = obj2_id
            else:
                # obj2 is the mocap, obj1 is the welded body
                mocap_id = self.physics.model.body_mocapid[obj2_id]
                body_idx = obj1_id
            #print(mocap_id)
            if mocap_id == -1 :
                continue
            self.physics.data.mocap_pos[mocap_id][:] = self.physics.model.body_pos[body_idx]
            self.physics.data.mocap_quat[mocap_id][:] = self.physics.model.body_quat[body_idx]

    def _set_action(self, ctrl):
        if self.action_type == 'torque':
            if self.use_dof == 'both':
                self.physics.data.ctrl[:] = ctrl
            elif self.use_dof == 'arm':
                self.physics.data.ctrl[self.action_arm_inds] = ctrl
            else:
                self.physics.data.ctrl[self.action_gripper_inds] = ctrl + 1
        elif self.action_type == 'velocity':
            self.physics.data.ctrl[:] = 0
            if self.use_dof == 'both':
                self.physics.data.qvel[0:len(ctrl)] = ctrl
            elif self.use_dof == 'arm':
                self.physics.data.qvel[self.action_arm_inds] = ctrl
            else:
                self.physics.data.qvel[self.action_gripper_inds] = ctrl
        else:
            self.physics.data.ctrl[:] = 0
            #self.reset_mocap2body_xpos()
            self.physics.data.mocap_pos[0, 0:len(ctrl)] = ctrl

    def get_current_info(self):
        return {}

    # Env specific helper functions
    # ----------------------------
    def get_achieved_goal_state(self):
        ref_pose = self.physics.data.qpos[:7]
        thetas = self.physics.data.qpos[self.state_rope_inds[7:]]
        return np.hstack([ref_pose, np.cos(thetas), np.sin(thetas)])

    def configure_indexes(self):
        # Arm and gripper action ind
        list_joints = get_name_arr_and_len(self.physics.named.data.ctrl, 0)[0]
        self.action_arm_inds = [idx for idx, s in enumerate(list_joints) if 'tj' in s]
        self.action_gripper_inds = [idx for idx, s in enumerate(list_joints) if 'tg' in s]

        # Rope
        list_qpos = get_name_arr_and_len(self.physics.named.data.qpos, 0)[0]

        self.state_arm_inds = [idx for idx, s in enumerate(list_qpos) if 'arm_' in s]
        self.state_gripper_inds = [idx for idx, s in enumerate(list_qpos) if 'gripper' in s]
        self.state_rope_ref_inds = [idx for idx, s in enumerate(list_qpos) if s == "J_ref"]
        self.state_rope_rot_inds = [idx for idx, s in enumerate(list_qpos) if s[0] == "J" and s != 'J_ref']
        self.state_rope_inds = self.state_rope_ref_inds + self.state_rope_rot_inds

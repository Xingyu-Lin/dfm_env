# Created by Xingyu Lin, 2018/11/18                                                                                  
import gym
import numpy as np

from .base import Base
from .utils.util import get_name_arr_and_len
from dm_control import mujoco
from dm_control.suite import base
import random


class RopeEnv(Base, gym.utils.EzPickle):
    def __init__(self, model_path='tasks/rope_temp.xml', distance_threshold=1e-2, distance_threshold_obs=0,
                 n_substeps=20, horizon=200, image_size=400, action_type='mocap',
                 with_goal=False,
                 use_visual_observation=True,
                 use_image_goal=True,
                 use_true_reward=False, use_dof='both', fix_gripper=True):
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
                      distance_threshold_obs=distance_threshold_obs, use_true_reward=use_true_reward)

        gym.utils.EzPickle.__init__(self)

    # Implementation of functions from GoalEnvExt
    # ----------------------------
    def _init_configure(self):

        self.configure_indexes()
        n1 = len(self.state_arm_inds)
        n2 = len(self.state_gripper_inds)
        n3 = len(self.state_rope_rot_inds)
        init_state_rope_ref = [-0.15, 0.5, 0.92, 1, 0, 0, 0]

        self.gripper_init_pos = self._sample_rope_init_pos()

        self.gripper_init_quat = [0, 1, 0, 0]
        self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))

        self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)

        init_arm_qpos = self.physics.data.qpos[self.state_arm_inds]
        # init_gripper_qpos = self.physics.data.qpos[self.state_gripper_inds]
        init_gripper_qpos = np.zeros((2,))
        self.init_qpos = np.hstack([init_arm_qpos, init_gripper_qpos, init_state_rope_ref, np.zeros(n3)])

        self.init_qvel = np.zeros(len(self.physics.data.qvel), )

    def _reset_sim(self):
        # Sample goal and render image

        with self.physics.reset_context():
            self.physics.data.qpos[:] = self.init_qpos
            self.physics.data.qvel[:] = self.init_qvel
            self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))
            self.gripper_init_pos = self._sample_rope_init_pos()
            self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)

        # Get the goal after the environment is stable
        self._random_rope_motion()

        self.goal_state = self.get_achieved_goal_state()

        if self.use_image_goal:
            self.goal_observation = self.render(depth=False)

        with self.physics.reset_context():
            self.physics.data.qpos[:] = self.init_qpos
            self.physics.data.qvel[:] = self.init_qvel
            self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))
            self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)

        return True

    def _sample_goal_state(self):
        """Samples a new goal in state space and returns it.
        """
        n = len(self.state_rope_rot_inds)
        thetas = np.random.random(n, ) * 0.1

        goal_state = np.hstack([self.init_qpos[self.state_rope_ref_inds], np.cos(thetas), np.sin(thetas)])
        return goal_state, thetas

    def _random_rope_motion(self):
        """Samples a random rope motor and physics steps to get new goal position.
        """
        # TODO apply force instead of torque
        #Sample a random rope actautor and action. Take 40 environment steps
        random_index = self.ctrl_rope_indices[random.randrange(len(self.ctrl_rope_indices))]
        self.physics.data.ctrl[random_index] += np.random.random(1, )
        for _ in range(30):
            self.physics.step()

        # To get a stable rope configuration after gravity takes effect.
        self.physics.data.ctrl[:] = 0
        for _ in range(1000):
            self.physics.step()

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

    def rope_control(self, idx, ctrl):
        self.physics.data.ctrl[idx] = ctrl

    def _apply_mocap_boundary(self, ctrl):

        boundary_range = [[-0.61, 0.5], [0.12, 0.85], [0.84, 0.88]]  # [min_val, max_val] for each of the dimension
        for i in range(3):
            if self.physics.data.mocap_pos[0][i] + ctrl[i] < boundary_range[i][0] and ctrl[i] < 0:
                ctrl[i] = 0
            elif self.physics.data.mocap_pos[0][i] + ctrl[i] > boundary_range[i][1] and ctrl[i] > 0:
                ctrl[i] = 0
        return ctrl

    def _set_action(self, ctrl):
        # todo ADD argument for mocap constrained 2d
        ctrl /= 20
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
            # ctrl[2] = 0
            # if self.time_step % 50 == 0:
            #     self.reset_mocap2body_xpos()
            self.physics.data.mocap_quat[:] = self.gripper_init_quat
            self.physics.data.mocap_pos[0, :3] += ctrl[:3]
            if not self.fix_gripper:
                self.physics.data.ctrl[self.ctrl_gripper_indices] = ctrl[3:]
            else:
                self.physics.named.data.qpos[self.state_gripper_inds] = 0
            # print(self._distance_between_gripper_rope_ref())
            # if self._distance_between_gripper_rope_ref() > 0.4 :
            #     self.gripper_init_pos = self._sample_rope_init_pos()
            #     self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)

    # Env specific helper functions
    # ----------------------------
    def _sample_rope_init_pos(self):
        # Sample one of the rope elements and put the gripper on top of it
        list_xpos = get_name_arr_and_len(self.physics.named.data.xpos, 0)[0]
        rope_xpos_inds = [idx for idx, s in enumerate(list_xpos) if s[0] == 'B']
        sampled_idx = np.random.choice(rope_xpos_inds, 1)
        return self.physics.data.xpos[sampled_idx] + np.array([0, 0, 0.2])

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

        list_ctrl = get_name_arr_and_len(self.physics.named.data.ctrl, 0)[0]

        self.ctrl_arm_indices = [idx for idx, s in enumerate(list_ctrl) if 'tj' in s]
        self.ctrl_gripper_indices = [idx for idx, s in enumerate(list_ctrl) if 'tg' in s]
        self.ctrl_rope_indices = [idx for idx, s in enumerate(list_ctrl) if 'tr' in s]

    def _move_gripper(self, gripper_target, gripper_rotation):
        self.physics.data.mocap_pos[0][:] = gripper_target
        self.physics.data.mocap_quat[0][:] = gripper_rotation
        for _ in range(500):
            self.physics.step()

    def _distance_between_gripper_rope_ref(self):
        return np.linalg.norm(self.physics.named.data.xpos['B7'] - self.physics.named.data.xpos['arm_gripper_base'],
                              axis=-1)

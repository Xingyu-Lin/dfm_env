# Created by Xingyu Lin, 2018/12/6
import numpy as np
from dfm_env.sawyer_env import SawyerEnv
from .utils.util import get_name_arr_and_len


class BlockPushEnv(SawyerEnv):
    def __init__(self, distance_threshold=5e-2, **kwargs):
        super(BlockPushEnv, self).__init__(model_path='tasks/block.xml', distance_threshold=distance_threshold, **kwargs)

    # Implementation of functions from GoalEnvExt
    # ----------------------------
    def _get_obs(self):
        if self.use_visual_observation:
            obs = self.render(depth=False)
        else:
            block_pos = self._get_block_pos()
            obs = np.concatenate((self.physics.data.qpos[self.state_arm_inds].copy(),
                                  self.physics.data.qpos[self.state_gripper_inds].copy(),
                                  block_pos, self.physics.data.qvel.copy()), axis=0)

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

    def _init_configure(self):
        self.boundary_range = [[-0.61, 0.5], [0.12, 0.85], [0.75, 1.0]]  # [min_val, max_val] for each of the dimension
        self.configure_indexes()
        n1 = len(self.state_arm_inds)
        n2 = len(self.state_gripper_inds)
        init_state_block_all = [-0.15, 0.5, 0.705, 1, 0, 0, 0]

        self.gripper_init_pos = [-0.15, 0.5, 0.905]  # Fixed initial position

        self.gripper_init_quat = [0, 1, 0, 0]
        self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))

        self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)

        init_arm_qpos = self.physics.data.qpos[self.state_arm_inds]
        # init_gripper_qpos = self.physics.data.qpos[self.state_gripper_inds]
        init_gripper_qpos = np.zeros((2,))
        self.init_qpos = np.hstack([init_arm_qpos, init_gripper_qpos, init_state_block_all])

        self.init_qvel = np.zeros(len(self.physics.data.qvel), )

    def _reset_sim(self):
        # Sample goal and render image
        with self.physics.reset_context():
            self.physics.data.qpos[:] = self.init_qpos
            self.physics.data.qvel[:] = self.init_qvel
            self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))
            self.gripper_init_pos = self.gripper_init_pos
            self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)

        # Get the goal after the environment is stable
        self.goal_state, _ = self._sample_goal_state()

        if self.use_image_goal:
            self._set_block_pos(self.goal_state)
            self.goal_observation = self.render(depth=False)

        with self.physics.reset_context():
            self.physics.data.qpos[:] = self.init_qpos
            self.physics.data.qvel[:] = self.init_qvel
            self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))

            self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)
            self._set_target_pos(self.goal_state)
        init_block_pos = self._sample_block_pos()
        self._set_block_pos(init_block_pos)
        return True

    def _sample_goal_state(self):
        """Samples a new goal in state space and returns it.
        """
        return self._sample_block_pos(), None

    # Env specific helper functions
    # ----------------------------

    def _sample_block_pos(self):
        boundary_range = [[-0.41, 0.3], [0.20, 0.60], [0.72498775, 0.72498775]]
        boundary_range = list(map(list, zip(*boundary_range)))

        block_pos = np.random.uniform(boundary_range[0], boundary_range[1])
        return block_pos

    def _set_block_pos(self, block_pos):
        self.physics.data.qpos[self.state_block_inds] = block_pos

    def _set_target_pos(self, block_pos):
        self.physics.named.model.body_pos[self.state_goal_body_ind] = block_pos

    def _get_block_pos(self):
        return self.physics.data.qpos[self.state_block_inds]

    def get_achieved_goal_state(self):
        return self._get_block_pos()

    def configure_indexes(self):
        # Arm and gripper action ind
        list_joints = get_name_arr_and_len(self.physics.named.data.ctrl, 0)[0]
        self.action_arm_inds = [idx for idx, s in enumerate(list_joints) if 'tj' in s]
        self.action_gripper_inds = [idx for idx, s in enumerate(list_joints) if 'tg' in s]

        # block
        list_qpos = get_name_arr_and_len(self.physics.named.data.qpos, 0)[0]

        self.state_arm_inds = [idx for idx, s in enumerate(list_qpos) if 'arm_' in s]
        self.state_gripper_inds = [idx for idx, s in enumerate(list_qpos) if 'gripper' in s]
        self.state_block_all_inds = [idx for idx, s in enumerate(list_qpos) if 'j_block' in s]
        self.state_block_inds = self.state_block_all_inds[:3]  # Only the pose and ignore the position

        list_site_xpos = get_name_arr_and_len(self.physics.named.model.body_pos, 0)[0]

        # print(self.physics.named.model.site_pos)
        self.state_goal_body_ind = [idx for idx, s in enumerate(list_site_xpos) if s == 'target'][0]

        list_ctrl = get_name_arr_and_len(self.physics.named.data.ctrl, 0)[0]

        self.ctrl_arm_indices = [idx for idx, s in enumerate(list_ctrl) if 'tj' in s]
        self.ctrl_gripper_indices = [idx for idx, s in enumerate(list_ctrl) if 'tg' in s]

    # def _distance_between_gripper_rope_ref(self):
    #     return np.linalg.norm(self.physics.named.data.xpos['B7'] - self.physics.named.data.xpos['arm_gripper_base'],
    #                           axis=-1)

# Created by Xingyu Lin, 2018/12/6
import numpy as np
from dfm_env.sawyer_env import SawyerEnv
from .utils.util import get_name_arr_and_len


class BlockPushEnv(SawyerEnv):
    def __init__(self, **kwargs):
        super(BlockPushEnv, self).__init__(model_path='tasks/block.xml')

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

    # Implementation of functions from GoalEnvExt
    # ----------------------------
    def _init_configure(self):

        self.configure_indexes()
        n1 = len(self.state_arm_inds)
        n2 = len(self.state_gripper_inds)
        n3 = len(self.state_rope_rot_inds)
        init_state_rope_ref = [-0.15, 0.5, 0.705, 1, 0, 0, 0]

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
        # Option 1:
        # self._random_rope_motion()
        # Option 2:
        self.goal_state, goal_theta = self._sample_goal_state()
        with self.physics.reset_context():
            # Move gripper to somewhere faraway from the rope
            self._move_gripper(gripper_target=[-0.80443307, 1.11125423, 1.08857343],
                               gripper_rotation=self.gripper_init_quat)
            self.physics.data.qpos[self.state_rope_rot_inds] = goal_theta
            self.physics.data.qpos[self.state_rope_ref_inds[2]] += 0.2
            for _ in range(200):
                self.physics.step()
            self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)

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
        thetas = (np.random.random(n, ) - 0.5) * 2
        flexible_joints = np.random.randint(4, size=n)
        flexible_joints = np.maximum(flexible_joints - 2, 0)
        thetas = thetas * flexible_joints
        goal_state = np.hstack([self.init_qpos[self.state_rope_ref_inds], np.cos(thetas), np.sin(thetas)])
        return goal_state, thetas

    def _random_rope_motion(self):
        """Samples a random rope motor and physics steps to get new goal position.
        """
        # TODO apply force instead of torque
        # Sample a random rope actautor and action. Take 40 environment steps
        random_index = self.ctrl_rope_indices[random.randrange(len(self.ctrl_rope_indices))]
        self.physics.data.ctrl[random_index] += np.random.random(1, )
        for _ in range(30):
            self.physics.step()

        # To get a stable rope configuration after gravity takes effect.
        self.physics.data.ctrl[:] = 0
        for _ in range(1000):
            self.physics.step()

    # Env specific helper functions
    # ----------------------------
    def rope_control(self, idx, ctrl):
        self.physics.data.ctrl[idx] = ctrl

    def _sample_rope_init_pos(self):
        # Sample one of the rope elements and put the gripper on top of it
        list_xpos = get_name_arr_and_len(self.physics.named.data.xpos, 0)[0]
        rope_xpos_inds = [idx for idx, s in enumerate(list_xpos) if s[0] == 'B']
        sampled_idx = np.random.choice(rope_xpos_inds, 1)
        return self.physics.data.xpos[sampled_idx] + np.array([0, 0, -0.05])

    def get_achieved_goal_state(self):
        ref_pose = self.physics.data.qpos[3:7]
        thetas = self.physics.data.qpos[self.state_rope_inds[7:]]
        # return np.hstack([ref_pose, np.cos(thetas), np.sin(thetas)])
        # Only need to achieve the angles
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

    def _distance_between_gripper_rope_ref(self):
        return np.linalg.norm(self.physics.named.data.xpos['B7'] - self.physics.named.data.xpos['arm_gripper_base'],
                              axis=-1)

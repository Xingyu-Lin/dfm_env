# Created by Xingyu Lin, 2018/12/6
import numpy as np
from dfm_env.sawyer_env import SawyerEnv
from dfm_env.utils.util import get_name_arr_and_len, ignored_physics_warning, cv_render


class RopeEnv(SawyerEnv):
    def __init__(self, distance_threshold=5e-2, fix_gripper=False,
                 rope_goal_sample_strategy='random_push', **kwargs):
        if fix_gripper:
            model_path = 'tasks/rope_temp.xml'
        else:
            model_path = 'tasks/rope_gripper_temp.xml'
        assert rope_goal_sample_strategy in ['random_drop', 'random_push', 'random_rope_motor']
        self.rope_goal_sample_strategy = rope_goal_sample_strategy
        super(RopeEnv, self).__init__(model_path=model_path, distance_threshold=distance_threshold,
                                      fix_gripper=fix_gripper, **kwargs)

    def _get_obs(self):
        if self.use_visual_observation:
            obs = self.render(depth=False)
        else:
            # thetas = self.physics.data.qpos[self.state_rope_rot_inds]
            obs = np.concatenate((self.physics.data.qpos[self.state_arm_inds].copy(),
                                  self.physics.data.qpos[self.state_gripper_inds].copy(),
                                  self.physics.data.qpos[self.state_rope_ref_inds].copy(),
                                  self.physics.data.qpos[self.state_rope_rot_inds].copy(),
                                  self.physics.data.qvel.copy()), axis=0)

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
        self.boundary_range = [[-0.61, 0.5], [0.12, 0.85],
                               [0.8405, 0.88]]  # [min_val, max_val] for each of the dimension
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
        if self.fix_gripper:
            init_gripper_qpos = np.zeros((0,))
        else:
            init_gripper_qpos = np.zeros((2,))
        self.init_indexes = self.state_arm_inds + self.state_gripper_inds + self.state_rope_ref_inds + self.state_rope_rot_inds
        self.init_qpos = np.hstack([init_arm_qpos, init_gripper_qpos, init_state_rope_ref, np.zeros(n3)])
        self.init_qvel = np.zeros(len(self.init_indexes), )
        # print(len(self.state_arm_inds), len(init_arm_qpos))
        # print(len(init_gripper_qpos), len(self.state_gripper_inds))
        # print(len(init_state_rope_ref), len(self.state_rope_ref_inds))
        # print(n3, len(self.state_rope_rot_inds))
        # exit()

    def _reset_sim(self):
        # Sample goal and render image, Get the goal after the environment is stable

        with self.physics.reset_context():
            self.physics.data.qpos[self.init_indexes] = self.init_qpos
            self.physics.data.qvel[self.init_indexes] = self.init_qvel
            self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))
            self.gripper_init_pos = self._sample_rope_init_pos()
            self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)

            if self.rope_goal_sample_strategy == 'random_rope_motor':
                self._random_rope_motion()
            elif self.rope_goal_sample_strategy == 'random_drop':
                self.goal_state, goal_theta = self._sample_goal_state()
                # Use the target rope
                self.physics.data.qpos[self.state_target_rope_ref_inds] = self.physics.data.qpos[
                    self.state_rope_ref_inds]
                self.physics.data.qpos[self.state_target_rope_ref_inds[1]] -= self.visualization_offset
                self.physics.data.qpos[self.state_target_rope_ref_inds[2]] += 0.2
                self.physics.data.qpos[self.state_target_rope_rot_inds] = goal_theta
                for _ in range(1000):
                    self.physics.step()
            elif self.rope_goal_sample_strategy == 'random_push':
                # return True
                self._random_push_rope()
                # self.gripper_init_pos = self._sample_rope_init_pos()
                # self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)

            if self.use_image_goal or True:
                self.physics.data.qpos[self.state_target_rope_ref_inds[1]] += self.visualization_offset
                target_original_transparancy = self.physics.model.geom_rgba[self.target_rope_geom_rgba_inds, 3][0]
                self.physics.model.geom_rgba[self.target_rope_geom_rgba_inds, 3] = 1.
                self.physics.model.geom_rgba[self.rope_geom_rgba_inds, 3] = 0

                # self.physics.model.geom_rgba[1, :] = np.asarray([0., 0., 0, 0.])  # Make the goal transparent
                self.goal_observation = self.render(depth=False)
                self.physics.data.qpos[self.state_target_rope_ref_inds[1]] -= self.visualization_offset
                self.physics.model.geom_rgba[self.target_rope_geom_rgba_inds, 3] = target_original_transparancy
                self.physics.model.geom_rgba[self.rope_geom_rgba_inds, 3] = 1.
                # Set the target qpos
                # self.physics.data.qpos[self.state_target_rope_inds] = self.physics.data.qpos[self.state_rope_inds]
                # self.physics.data.qvel[self.state_target_rope_inds] = 0
        self.goal_state = self.get_target_goal_state()
        # Move gripper to somewhere faraway from the rope
        # self._move_gripper(gripper_target=[-0.80443307, 1.11125423, 1.08857343],
        #                    gripper_rotation=self.gripper_init_quat)

        # with self.physics.reset_context():
        #     self.physics.data.qpos[self.init_indexes] = self.init_qpos
        #     self.physics.data.qvel[self.init_indexes] = self.init_qvel
        #     self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))
        #     self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)

        return True

    def _sample_goal_state(self):
        """Samples a new goal in state space and returns it.
        """
        n = len(self.state_rope_rot_inds)
        thetas = (np.random.random(n, ) - 0.5) * 2
        flexible_joints = np.random.randint(4, size=n)
        flexible_joints = np.maximum(flexible_joints - 2, 0)
        thetas = thetas * flexible_joints
        goal_state = np.hstack([np.cos(thetas), np.sin(thetas)])
        return goal_state, thetas

    # Env specific helper functions
    # ----------------------------
    def rope_control(self, idx, ctrl):
        self.physics.data.ctrl[idx] = ctrl

    def get_target_goal_state(self):
        # thetas = self.physics.data.qpos[self.state_target_rope_inds[7:]]
        # return np.hstack([np.cos(thetas), np.sin(thetas)])
        return self.physics.data.xpos[self.target_rope_xpos_inds, :2].flatten().copy()

    def get_achieved_goal_state(self):
        # ref_pose = self.physics.data.qpos[self.state_rope_ref_inds[-4:]]
        # thetas = self.physics.data.qpos[self.state_rope_inds[7:]]
        # return np.hstack([ref_pose, np.cos(thetas), np.sin(thetas)])
        # Only need to achieve the angles
        # return np.hstack([np.cos(thetas), np.sin(thetas)])
        return self.physics.data.xpos[self.rope_xpos_inds, :2].flatten().copy()

    # noinspection PyAttributeOutsideInit
    def configure_indexes(self):
        # Arm and gripper action joints
        list_joints = get_name_arr_and_len(self.physics.named.data.ctrl, 0)[0]
        self.action_arm_inds = [idx for idx, s in enumerate(list_joints) if 'tj' in s]
        self.action_gripper_inds = [idx for idx, s in enumerate(list_joints) if 'tg' in s]

        # qpos index for arm, gripper and rope
        list_qpos = get_name_arr_and_len(self.physics.named.data.qpos, 0)[0]
        self.state_arm_inds = [idx for idx, s in enumerate(list_qpos) if 'arm_' in s]
        self.state_gripper_inds = [idx for idx, s in enumerate(list_qpos) if 'gripper' in s]
        self.state_rope_ref_inds = [idx for idx, s in enumerate(list_qpos) if s == "Rope_ref"]
        self.state_rope_rot_inds = [idx for idx, s in enumerate(list_qpos) if s.startswith('Rope_J')]
        self.state_rope_inds = self.state_rope_ref_inds + self.state_rope_rot_inds
        self.state_target_rope_ref_inds = [idx for idx, s in enumerate(list_qpos) if s == "targetRope_ref"]
        self.state_target_rope_rot_inds = [idx for idx, s in enumerate(list_qpos) if s.startswith('targetRope_J')]
        self.state_target_rope_inds = self.state_target_rope_ref_inds + self.state_target_rope_rot_inds
        self.state_push_block_inds = [idx for idx, s in enumerate(list_qpos) if 'push_block_slide' in s]
        self.state_push_block_inds = self.state_push_block_inds[:2]

        list_ctrl = get_name_arr_and_len(self.physics.named.data.ctrl, 0)[0]
        self.ctrl_arm_inds = [idx for idx, s in enumerate(list_ctrl) if 'tj' in s]
        self.ctrl_gripper_inds = [idx for idx, s in enumerate(list_ctrl) if 'tg' in s]
        self.ctrl_rope_inds = [idx for idx, s in enumerate(list_ctrl) if 'tr' in s]

        list_geom = get_name_arr_and_len(self.physics.named.model.geom_rgba, 0)[0]
        self.rope_geom_rgba_inds = [idx for idx, s in enumerate(list_geom) if s != 0 and s.startswith('Rope_G')]
        self.target_rope_geom_rgba_inds = [idx for idx, s in enumerate(list_geom) if
                                           s != 0 and s.startswith('targetRope_G')]
        self.push_block_geom_rgba_inds = [idx for idx, s in enumerate(list_geom) if
                                          s != 0 and s.startswith('pushBlock_G')]

        list_xpos = get_name_arr_and_len(self.physics.named.data.xpos, 0)[0]
        self.rope_xpos_inds = [idx for idx, s in enumerate(list_xpos) if s.startswith('Rope_')]
        self.target_rope_xpos_inds = [idx for idx, s in enumerate(list_xpos) if s.startswith('targetRope_')]
        self.visualization_offset = 0.1

    # Helper functions for sampling goals
    # ------------------------------

    def _distance_between_gripper_rope_ref(self):
        return np.linalg.norm(
            self.physics.named.data.xpos['Rope_B7'] - self.physics.named.data.xpos['arm_gripper_base'],
            axis=-1)

    def _sample_rope_init_pos(self):
        # Sample one of the rope elements and put the gripper on top of it
        sampled_idx = np.random.choice(self.rope_xpos_inds, 1)
        return self.physics.data.xpos[sampled_idx] + np.array([0, 0, 0.2])

    def _sample_rope_neighbourhood(self):
        # Sample one of the rope elements and put the gripper on top of it
        sampled_idx = np.random.choice(self.rope_xpos_inds, 1)
        point_start = self.physics.data.xpos[sampled_idx][0].copy()
        point_end = point_start.copy()
        delta_x = (np.random.random() - 0.5) / 10
        delta_y = np.sign(np.random.random() - 0.5) * 0.03
        delta_xy = [delta_x, delta_y]
        point_start[:2] += delta_xy
        point_end[:2] -= delta_xy
        # Only need this if the sampled neighbourhood is for the sawyer gripper
        # point_start[2] = self.boundary_range[2][0]  # Account for the height of the gripper
        # point_end[2] = self.boundary_range[2][0]  # Account for the height of the gripper
        return point_start, point_end

    def _random_rope_motion(self):
        """Samples a random rope motor and physics steps to get new goal position.
        """
        # TODO apply force instead of torque
        # Sample a random rope actautor and action. Take 40 environment steps
        random_index = self.ctrl_rope_inds[random.randrange(len(self.ctrl_rope_inds))]
        self.physics.data.ctrl[random_index] += np.random.random(1, )
        for _ in range(30):
            self.physics.step()

        # To get a stable rope configuration after gravity takes effect.
        self.physics.data.ctrl[:] = 0
        for _ in range(1000):
            self.physics.step()

    # This was using the sawyer gripper to give one push to the rope. We now use a block to do the pushing to make it
    # more stable
    # def _random_push_rope(self):
    #     # Sample two points near the rope and let the gripper to push from one point to another point
    #     # Keep pushing until the position of the rope changes
    #     # init_rope_qpos = self.physics.data.qpos[self.state_rope_inds]
    #     init_rope_state = self.get_achieved_goal_state()
    #     while True:
    #         start_pt, end_pt = self._sample_rope_neighbourhood()
    #         # Account for the height of the gripper
    #         self._move_gripper(start_pt, render=True, verbose=True)
    #         # end_pt[2] = self.physics.named.data.xpos['arm_gripper_base'][2]
    #         self._move_gripper(end_pt, render=True, verbose=True)
    #         print('two points:', start_pt, end_pt)
    #         with ignored_physics_warning():
    #             for _ in range(100):
    #                 self.physics.step()  # Wait for rope to stablize
    #         cur_rope_state = self.get_achieved_goal_state()
    #         distance = np.linalg.norm(init_rope_state - cur_rope_state)
    #         print(distance)
    #         if 1e-1 < distance < 3:
    #             # Set the target rope qpos and reset the rope qpos
    #             # self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)
    #             target_rope_qpos = self.physics.data.qpos[self.state_rope_inds].copy()
    #             self.physics.data.qpos[self.init_indexes] = self.init_qpos
    #             self.physics.data.qvel[self.init_indexes] = self.init_qvel
    #             self.physics.data.qpos[self.state_target_rope_inds] = target_rope_qpos
    #             # self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)
    #             self._move_gripper(gripper_target=end_pt, gripper_rotation=self.gripper_init_quat)
    #             return
    #         else:
    #             self.physics.data.qpos[self.init_indexes] = self.init_qpos
    #             self.physics.data.qvel[self.init_indexes] = self.init_qvel

    def _get_push_block_pos(self):
        return self.physics.data.qpos[self.state_gripper_inds]

    def _set_push_block_pos(self, target_pos):
        self.physics.data.qpos[self.state_gripper_inds] = target_pos
        return

    def _set_push_block_vel(self, target_vel):
        self.physics.data.qvel[self.state_gripper_inds] = target_vel

    def _move_push_block(self, start_pt, end_pt, render=False):
        # if render:
        #     self.physics.model.geom_rgba[self.push_block_geom_rgba_inds, 3] = 1.
        self._set_push_block_pos(start_pt[:2])
        for _ in range(10):  # Let the block fall down if possible
            self.physics.step()
        move_step = 100
        target_vel = (end_pt - start_pt) * 20
        self._set_push_block_vel(target_vel[:2])
        for _ in range(move_step):
            self.physics.step()
            if render:
                img = self.physics.render(camera_id='static_camera')
                cv_render(img, name='push_block_display')
        self._set_push_block_vel([0., 0.])
        # Move the block away and set it to be invisible after usage
        self._set_push_block_pos([10., 10., ])
        self.physics.model.geom_rgba[self.push_block_geom_rgba_inds, 3] = 0.

    def _random_push_rope(self):
        # Sample two points near the rope and let the block to move from one point to another point
        # Keep trying until the position of the rope changes
        init_rope_state = self.get_achieved_goal_state()
        while True:
            start_pt, end_pt = self._sample_rope_neighbourhood()
            self._move_push_block(start_pt, end_pt, render=False)

            # with ignored_physics_warning():
            #     for _ in range(100):
            #         self.physics.step()  # Wait for rope to stablize
            cur_rope_state = self.get_achieved_goal_state()
            distance = np.linalg.norm(init_rope_state - cur_rope_state)
            print(distance)
            if 1e-5 < distance < 3:
                # Set the target rope qpos and reset the rope qpos
                # self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)
                target_rope_qpos = self.physics.data.qpos[self.state_rope_inds].copy()
                self.physics.data.qpos[self.init_indexes] = self.init_qpos
                self.physics.data.qvel[self.init_indexes] = self.init_qvel
                self.physics.data.qpos[self.state_target_rope_inds] = target_rope_qpos
                self._move_gripper(gripper_target=self.gripper_init_pos, gripper_rotation=self.gripper_init_quat)
                return
            else:
                self.physics.data.qpos[self.init_indexes] = self.init_qpos
                self.physics.data.qvel[self.init_indexes] = self.init_qvel

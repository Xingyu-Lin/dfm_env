# Created by Xingyu Lin, 2018/12/6
import gym
import numpy as np

from .base import Base
from .utils.util import get_name_arr_and_len, ignored_physics_warning, cv_render
from shapely.geometry import LineString, Point

class SawyerFloatEnv(Base, gym.utils.EzPickle):
    def __init__(self, model_path, distance_threshold, distance_threshold_obs=0,
                 n_substeps=20, horizon=200, image_size=400, action_type='endpoints',
                 with_goal=False,
                 use_visual_observation=True,
                 use_image_goal=True,
                 use_true_reward=False,
                 arm_height=0.88,
                 arm_move_velocity=0.4,
                 **kwargs):
        '''

        :param model_path:
        :param distance_threshold:
        :param distance_threshold_obs:
        :param n_substeps:
        :param horizon:
        :param image_size:
        :param action_type:
            'endpoints': The agent will specify the starting and the ending point of a push (Both in 2D)
        :param with_goal:
        :param use_visual_observation:
        :param use_image_goal:
        :param use_true_reward:
        :param use_dof: ['both', 'arm', 'gripper']
        Base class for sawyer manipulation environments
        '''
        self.action_type = action_type
        self.arm_move_velocity = arm_move_velocity
        self.arm_height = arm_height

        if action_type == 'endpoints' or action_type == 'pick_and_place':
            n_actions = 4
            self.prev_action_finished = True
        elif action_type == 'velocity':
            n_actions = 3
        else:
            assert NotImplementedError
        Base.__init__(self, model_path=model_path, n_substeps=n_substeps, horizon=horizon, n_actions=n_actions,
                      image_size=image_size, use_image_goal=use_image_goal,
                      use_visual_observation=use_visual_observation,
                      with_goal=with_goal, reward_type='sparse', distance_threshold=distance_threshold,
                      distance_threshold_obs=distance_threshold_obs, use_true_reward=use_true_reward, **kwargs)

        gym.utils.EzPickle.__init__(self)

    # TODO set the arm to be invisible
    # Implementation of functions from GoalEnvExt
    # ----------------------------
    def _set_action(self, ctrl):
        assert len(ctrl) == self.n_actions
        if self.action_type == 'velocity':
            # ctrl /= self.n_substeps
            self._set_arm_velocity(ctrl)
            return
        elif self.action_type == 'endpoints':
            point_st = self._transform_control(ctrl[:2])
            point_en = self._transform_control(ctrl[2:])
            point_st = np.append(point_st, self.arm_height)
            point_en = np.append(point_en, self.arm_height)
            self._move_arm_by_endpoints(point_st, point_en)
            return 'env_no_step'
        elif self.action_type == 'endpoints_visualization':
            if self.prev_action_finished:
                point_st = self._transform_control(ctrl[:2])
                point_en = self._transform_control(ctrl[2:])
                self.prev_point_st = np.append(point_st, self.arm_height)
                self.prev_point_en = np.append(point_en, self.arm_height)
            self._move_arm_by_endpoints_one_step(self.prev_point_st, self.prev_point_en)
            return 'env_no_step'
        elif self.action_type == 'pick_and_place':
            if self.prev_action_finished:
                point_st = self._transform_control(ctrl[:2])
                point_en = self._transform_control(ctrl[2:])
                self.prev_point_st = np.append(point_st, self.arm_height)
                self.prev_point_en = np.append(point_en, self.arm_height)
            self._pick_and_place_by_endpoints(self.prev_point_st, self.prev_point_en)
        elif self.action_type == 'pick_and_place_visualization':
            if self.prev_action_finished:
                point_st = self._transform_control(ctrl[:2])
                point_en = self._transform_control(ctrl[2:])
                self.prev_point_st = np.append(point_st, self.arm_height)
                self.prev_point_en = np.append(point_en, self.arm_height)
            self._pick_and_place_one_step(self.prev_point_st, self.prev_point_en)
            return 'env_no_step'

    def _get_obs(self):
        # print(self.get_arm_location())
        if self.use_visual_observation:
            obs = self.render(depth=False)
        else:
            # thetas = self.physics.data.qpos[self.state_rope_rot_inds]
            # obs = np.concatenate((self.physics.data.qpos[self.qpos_rope_ref_inds].copy(),
            #                       self.physics.data.qpos[self.qpos_rope_rot_inds].copy()), axis=0)
            obs = self.physics.data.xpos[self.xpos_rope_inds].copy()

        if self.use_image_goal:
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

    # noinspection PyAttributeOutsideInit
    def configure_indexes(self):
        # Arm and gripper action joints
        list_joints = get_name_arr_and_len(self.physics.named.data.ctrl, 0)[0]
        self.action_arm_inds = [idx for idx, s in enumerate(list_joints) if s.startswith('tj')]
        # No gripper for now; modify rope_float.xml later
        self.action_gripper_inds = [idx for idx, s in enumerate(list_joints) if 'gripper_joint_' in s]

        # qpos index for arm, gripper and rope
        list_qpos = get_name_arr_and_len(self.physics.named.data.qpos, 0)[0]
        self.qpos_arm_inds = [idx for idx, s in enumerate(list_qpos) if s.startswith('arm_slide')]
        self.qpos_gripper_inds = [idx for idx, s in enumerate(list_qpos) if 'gripper' in s]
        self.qpos_rope_ref_inds = [idx for idx, s in enumerate(list_qpos) if s == "Rope_ref"]
        self.qpos_rope_rot_inds = [idx for idx, s in enumerate(list_qpos) if s.startswith('Rope_J')]
        self.qpos_block_inds = [idx for idx, s in enumerate(list_qpos) if s.startswith('j_block')]
        self.qpos_rope_inds = self.qpos_rope_ref_inds + self.qpos_rope_rot_inds
        self.qpos_target_rope_ref_inds = [idx for idx, s in enumerate(list_qpos) if s == "targetRope_ref"]
        self.qpos_target_rope_rot_inds = [idx for idx, s in enumerate(list_qpos) if s.startswith('targetRope_J')]
        self.qpos_target_rope_inds = self.qpos_target_rope_ref_inds + self.qpos_target_rope_rot_inds
        self.qpos_gripblock_left_inds = [idx for idx, s in enumerate(list_qpos) if s.startswith("g1_block")]
        self.qpos_gripblock_right_inds = [idx for idx, s in enumerate(list_qpos) if s.startswith("g2_block")]

        # self.state_push_block_inds = [idx for idx, s in enumerate(list_qpos) if 'push_block_slide' in s]
        # self.state_push_block_inds = self.state_push_block_inds[:2]

        list_xfrc = get_name_arr_and_len(self.physics.named.data.xfrc_applied, 0)[0]
        self.rope_xfrc = [idx for idx, s in enumerate(list_xfrc) if s.startswith('Rope_')]
        self.arm_xfrc = [idx for idx, s in enumerate(list_xfrc) if s.startswith('arm_l6')]

        list_ctrl = get_name_arr_and_len(self.physics.named.data.ctrl, 0)[0]
        self.ctrl_arm_inds = [idx for idx, s in enumerate(list_ctrl) if 'tj' in s]
        self.ctrl_gripper_inds = [idx for idx, s in enumerate(list_ctrl) if 'tg' in s]
        self.ctrl_rope_inds = [idx for idx, s in enumerate(list_ctrl) if 'tr' in s]

        list_geom = get_name_arr_and_len(self.physics.named.model.geom_rgba, 0)[0]
        self.geom_rgba_rope_inds = [idx for idx, s in enumerate(list_geom) if s != 0 and s.startswith('Rope_G')]
        self.geom_rgba_target_rope_inds = [idx for idx, s in enumerate(list_geom) if
                                         s != 0 and s.startswith('targetRope_G')]
        #self.geom_rgba_rope_inds = [idx for idx, s in enumerate(list_geom) if s != 0 and s.startswith('block_G')]
        #self.geom_rgba_target_rope_inds = [idx for idx, s in enumerate(list_geom) if
        #                                   s != 0 and s.startswith('target_G')]
        self.geom_rgba_arm_inds = [idx for idx, s in enumerate(list_geom) if s == 0]
        # self.push_block_geom_rgba_inds = [idx for idx, s in enumerate(list_geom) if
        #                                   s != 0 and s.startswith('pushBlock_G')]

        list_xpos = get_name_arr_and_len(self.physics.named.data.xpos, 0)[0]
        self.xpos_arm_inds = [idx for idx, s in enumerate(list_xpos) if s == 'arm_l6']
        self.xpos_rope_inds = [idx for idx, s in enumerate(list_xpos) if s.startswith('Rope_')]
        self.xpos_finger = [idx for idx, s in enumerate(list_xpos) if 'finger' in s]
        self.ordered_rope_inds = []
        for i in range(0, len(self.xpos_rope_inds)):
            self.ordered_rope_inds.append([idx for idx, s in enumerate(list_xpos) if s.startswith('Rope_') and int(s.split('B')[1]) == i][0])

        self.xpos_target_rope_inds = [idx for idx, s in enumerate(list_xpos) if s.startswith('targetRope_')]
        #self.xpos_rope_inds = [idx for idx, s in enumerate(list_xpos) if s.startswith('block_')]
        #self.xpos_target_rope_inds = [idx for idx, s in enumerate(list_xpos) if s.startswith('target_')]
        self.visualization_offset = 0.1

        list_actuate = get_name_arr_and_len(self.physics.named.data.qfrc_applied, 0)[0]
        self.qfrc_actuate_gripper = [idx for idx, s in enumerate(list_actuate) if 'gripper' in s]
        self.qfrc_actuate_arm = [idx for idx, s in enumerate(list_actuate) if 'arm' in s]

        list_motors = get_name_arr_and_len(self.physics.named.data.ctrl,0)[0]
        self.grip_motors = [idx for idx, s in enumerate(list_motors) if 'tg0' in s]
        self.rope_motors = [idx for idx, s in enumerate(list_motors) if s.startswith('tr')]


        self.qfrc_actuate_block = [idx for idx, s in enumerate(list_actuate) if 'g1_block' in s]

        list_unc = get_name_arr_and_len(self.physics.named.data.qfrc_unc,0)[0]
        self.qfrc_block_unc = [idx for idx, s in enumerate(list_unc) if 'j_block' in s]

        weld_list = get_name_arr_and_len(self.physics.named.model.body_weldid, 0)[0]
        self.weld_block_weld = [idx for idx, s in enumerate(weld_list) if 'weld_block' in s]
        self.rope_weld = []
        for i in range(0, len(self.xpos_rope_inds)):
            self.rope_weld.append(
                [idx for idx, s in enumerate(list_xpos) if s.startswith('Rope_') and int(s.split('B')[1]) == i][0])

        self.block_weld = [idx for idx, s in enumerate(weld_list) if s.startswith('block')]

        constr_list = get_name_arr_and_len(self.physics.named.model.eq_obj1id, 0)[0]
        self.active_weld = [idx for idx, s in enumerate(constr_list) if s.startswith('weld1')]






    def _init_configure(self):
        # self.boundary_range = [[-0.8, 0.75], [-0.3, 0.83]]  # [min_val, max_val] for each of the dimension
        self.boundary_range = [[-0.56, 0.51], [-0.28, 0.80]]  # [min_val, max_val] for each of the dimension
        self.boundary_coeff = [(self.boundary_range[i][1] - self.boundary_range[i][0]) / 2. for i in
                               range(len(self.boundary_range))]
        self.configure_indexes()
        n1 = len(self.qpos_arm_inds)
        n2 = len(self.qpos_gripper_inds)
        n3 = len(self.qpos_rope_rot_inds)
        init_state_rope_ref = [0, 0.3, 0.705, 1, 0, 0, 0]

        self.init_arm_xpos = self._sample_rope_init_xpos()
        # self.arm_init_quat = [0, 1, 0, 0]
        self.physics.data.ctrl[:] = np.zeros(len(self.physics.data.ctrl))
        self.set_arm_location(self.init_arm_xpos)

        init_arm_qpos = self.init_arm_xpos.squeeze()
        self.init_qpos_indexes = self.qpos_arm_inds + self.qpos_rope_ref_inds + self.qpos_rope_rot_inds
        self.init_qpos = np.hstack([init_arm_qpos, init_state_rope_ref, np.zeros(n3)])

        self.init_qvel = np.zeros(len(self.init_qpos_indexes), )

    def _reset_all_to_init_pos(self):
        self.arm_stage = 0
        self.physics.data.qpos[self.init_qpos_indexes] = self.init_qpos
        self.physics.data.qvel[self.init_qpos_indexes] = self.init_qvel
        self.physics.forward()

    def _sample_rope_init_xpos(self):
        # Implemented by rope env
        pass

    def get_current_info(self):
        """
        :return: The true current state, 'ag_state', and goal state, 'g_state'
        """
        info = {
            'ag_state': self.get_achieved_goal_state(),
            'g_state': self.goal_state.copy()
        }
        return info

    # Floating Sawyer environemtn specific helper functions
    # --------------------------------------------------
    def _transform_control(self, ctrl):
        ret = []
        for i in range(len(ctrl)):
            ret.append(self.boundary_range[i][0] + (ctrl[i] + 1) * self.boundary_coeff[i])
        return np.array(ret)

    # def _apply_endpoint_boundary(self, point):
    #     for i in range(2):
    #         point[i] = np.clip(point[i], self.boundary_range[i][0], self.boundary_range[i][1])
    #     return point

    def get_arm_location(self):
        return self.physics.data.qpos[self.qpos_arm_inds]

    def set_arm_location(self, arm_location):
        self.physics.data.qpos[self.qpos_arm_inds] = arm_location

    def _set_arm_velocity(self, arm_velocity):
        self.physics.data.qvel[self.qpos_arm_inds] = arm_velocity

    def _set_arm_force(self, arm_force):
        self.physics.data.qfrc_applied[self.qfrc_actuate_arm] = arm_force

    def get_gripper_state(self):
        return self.physics.data.qpos[self.qpos_gripper_inds]

    def get_finger_location(self):
        return self.physics.data.xpos[self.xpos_finger]

    def weld_rope_offset(self, idx):
        self.physics.model.eq_obj2id[self.active_weld] = self.rope_weld[idx]
        self.physics.model.eq_active[self.active_weld] = 1

    def unweld_rope_offset(self):
        self.physics.model.eq_active[self.active_weld] = 0

    def weld_block(self):
        #self.physics.model.eq_obj2id[self.active_weld] = self.block_weld[0]
        self.physics.model.eq_active[self.active_weld] = 1

    def set_gripper_state(self, gripper_state):
        self.physics.data.qpos[self.qpos_gripper_inds] = gripper_state

    def _set_gripper_velocity(self, gripper_velocity):
        self.physics.data.qvel[self.qpos_gripper_inds] = gripper_velocity

    def _set_gripper_force(self, gripper_force):
        self.physics.data.ctrl[self.grip_motors] = gripper_force

    def get_rope_start_location(self):
        return self.physics.data.qpos[self.qpos_rope_ref_inds][0:3]

    def set_rope_start_location(self, pos):
        self.physics.data.qpos[self.qpos_rope_ref_inds[0:3]] = pos

    def get_block_location(self):
        return self.physics.data.qpos[self.qpos_block_inds][0:3]

    def set_block_location(self, pos):
        self.physics.data.qpos[self.qpos_block_inds[0:3]] = pos

    def get_grip_block_location(self):
        return self.physics.data.qpos[self.qpos_gripblock_left_inds][0:3]

    def set_grip_block_location(self, loc):
        self.physics.data.qpos[self.qpos_gripblock_left_inds[0:3]] = loc
        loc[1] = loc[1]+0.1
        self.physics.data.qpos[self.qpos_gripblock_right_inds[0:3]] = loc

    def set_grip_block_velocity(self, vel):
        self.physics.data.qvel[self.qpos_gripblock_left_inds][0:3] = vel
        self.physics.data.qvel[self.qpos_gripblock_right_inds][0:3] = vel

    def set_grip_block_vel_single(self, vel):
        self.physics.data.qvel[self.qpos_gripblock_left_inds[1]] = vel

    def set_grip_block_force(self, frc):
        self.physics.data.qfrc_applied[self.qfrc_actuate_block] = frc

    def get_grip_block_dist(self):
        return self.physics.data.qpos[self.qpos_gripblock_right_inds][1] - self.physics.data.qpos[self.qpos_gripblock_left_inds][1]

    def get_rope_offset_location(self, i):
        offset = self.physics.data.xpos[self.ordered_rope_inds[i]]
        return offset

    def set_rope_offset_location(self, i, pos):
        self.physics.data.xpos[self.xpos_rope_inds[i]] = pos

    def set_rope_offset_force(self, i, force):
        self.physics.data.xfrc_applied[self.rope_xfrc[i]] = force

    def set_rope_offset_force_partial(self, i, j, force):
        self.physics.data.xfrc_applied[self.rope_xfrc[i]][j] = force

    def get_arm_xfrc(self):
        return self.physics.data.xfrc_applied[self.arm_xfrc]
    @staticmethod
    def get_point_dist(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def _rope_intersection_filter(self, point_st, point_en, eps=0.07):
        # Take in a push for point_st to point_en, return a new push starting point
        line = LineString([(point_st[0], point_st[1]), (point_en[0], point_en[1])])
        intersection_pts = []
        for ind in self.xpos_rope_inds:
            if self.get_point_dist(point_st, self.physics.data.xpos[ind][:2]) < eps:  # start point inside the circle
                return point_st
            point = Point(*self.physics.data.xpos[ind][:2])  # Get the x,y postion of one point on the rope
            circle = point.buffer(eps).boundary
            intersection_pt = circle.intersection(line)
            intersection_pts.append(intersection_pt)
        best_dist = np.inf
        best_pt = None
        for pt in intersection_pts:
            if not isinstance(pt, Point):  # Do not count as intersection for boundary points
                for geom in pt.geoms:
                    dist = self.get_point_dist(point_st, geom.coords[0])
                    if dist < best_dist:
                        best_dist = dist
                        best_pt = geom.coords[0]
        if best_pt is not None:
            best_pt = [best_pt[0], best_pt[1], point_st[2]]

        return best_pt

    def perp(self, a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    def seg_intersect(self, a1, a2, b1, b2):
        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = self.perp(da)
        denom = np.dot(dap, db)
        if denom == 0:
            return None
        num1 = np.dot(dap, dp)
        num2 = np.dot(self.perp(db),dp)
        return (num1 / denom.astype(float)), (num2 / denom.astype(float))

    def get_closest_rope_point(self, l_finger_pos, r_finger_pos):

        finger_seg_1 = l_finger_pos[:2]
        finger_seg_2 = r_finger_pos[:2]
        finger_diff = r_finger_pos-l_finger_pos
        for i in range(0, len(self.ordered_rope_inds)-1):
            firstRopePos = self.get_rope_offset_location(i)
            secondRopePos = self.get_rope_offset_location(i+1)
            rope_diff = secondRopePos - firstRopePos
            easy_intersect = self.seg_intersect(finger_seg_1, finger_seg_2, firstRopePos[:2], secondRopePos[:2])
            if easy_intersect is not None:
                s, t = easy_intersect
                if 0 <= s <= 1 and 0 <= t <= 1:
                    return i, t*rope_diff+firstRopePos

    def _move_arm_by_endpoints(self, arm_st_loc, arm_en_loc, velocity=None, render=False, accelerated=True):
        # Move the arm from the start loc to the end loc
        # Accelearation: Only do the simulation when the earm intersects the arm
        if accelerated:
            arm_st_loc = self._rope_intersection_filter(arm_st_loc, arm_en_loc)
            if arm_st_loc is None:
                return
        if velocity is None:
            velocity = self.arm_move_velocity
        self.set_arm_location(arm_st_loc)
        move_dist = self.get_point_dist(arm_en_loc, arm_st_loc)
        if move_dist == 0:
            return
        move_dir = (arm_en_loc - arm_st_loc) / move_dist

        prev_dist = 10000
        cnt = 0

        while True:
            self._set_arm_velocity(move_dir * velocity)
            if render:
                img = self.physics.render(camera_id='static_camera')
                cv_render(img)
            with ignored_physics_warning():
                self.physics.step()
            cur_dist = self._get_endpoints_distance(self.get_arm_location(), arm_en_loc)
            if cur_dist > prev_dist:
                break
            prev_dist = cur_dist
            cnt += 1

    def _pick_and_place_by_endpoints(self, arm_st_loc, arm_en_loc, velocity=None, render=False,
                                     accelerated=True, init_height=1.5, grip_open=0.020833, grip_close=0.01 ):
        # pick up whatever is in the initial position, and place it at the end position

        if accelerated:
            arm_st_loc = self._rope_intersection_filter(arm_st_loc, arm_en_loc)
            if arm_st_loc is None:
                return
        if velocity is None:
            velocity = self.arm_move_velocity
        hovered = [arm_st_loc[0], arm_st_loc[1], init_height]
        self.set_arm_location(hovered)
        self.set_gripper_state(grip_open)
        self._set_gripper_velocity(0)

        # lower the arm

        down_dir = np.asarray([0, 0, -1])
        prev_dist = 10000
        cnt = 0

        while True:
            print("down\n")
            self._set_arm_velocity(down_dir * velocity)
            if render:
                img = self.physics.render(camera_id='static_camera')
                cv_render(img)
            with ignored_physics_warning():
                self.physics.step()
            cur_dist = self._get_endpoints_distance(self.get_arm_location(), arm_st_loc)
            if cur_dist > prev_dist:
                break
            prev_dist = cur_dist
            cnt += 1
            self.set_gripper_state(grip_open)


        # close the gripper
        # rope diam = 0.01
        target = grip_close
        gripper_vel = (target - self.get_gripper_state())
        prev_dist = 10000
        cnt = 0
        while True:
            print("close: {}, target: {}\n".format(self.get_gripper_state(), target))
            self._set_gripper_velocity(gripper_vel)
            if render:
                img = self.physics.render(camera_id='static_camera')
                cv_render(img)
            with ignored_physics_warning():
                self.physics.step()
            cur_dist = self.get_gripper_state() - target
            if self.get_gripper_state() < target:
                break
            prev_dist = cur_dist
            cnt += 1

        # raise arm
        """
        up_dir = np.asarray([0, 0, 1])
        prev_dist = 10000
        cnt = 0

        while True:
            print("up\n")
            self._set_arm_velocity(up_dir * velocity)
            if render:
                img = self.physics.render(camera_id='static_camera')
                cv_render(img)
            with ignored_physics_warning():
                self.physics.step()
            cur_dist = self._get_endpoints_distance(self.get_arm_location(), hovered)
            if cur_dist > prev_dist:
                break
            prev_dist = cur_dist
            cnt += 1
            #self.set_gripper_state(0.020833)
        """
        # move arm
        move_dist = self.get_point_dist(arm_en_loc, arm_st_loc)
        if move_dist == 0:
            return
        move_dir = (arm_en_loc - arm_st_loc) / move_dist
        prev_dist = 10000
        cnt = 0

        while True:
            print("move\n")
            self._set_arm_velocity(move_dir * velocity)
            if render:
                img = self.physics.render(camera_id='static_camera')
                cv_render(img)
            with ignored_physics_warning():
                self.physics.step()
            cur_dist = self._get_endpoints_distance(self.get_arm_location(), arm_en_loc)
            if cur_dist > prev_dist:
                break
            prev_dist = cur_dist
            cnt += 1
            #self.set_gripper_state(0.020833)

        # open gripper
        target = grip_close
        gripper_vel = (target - self.get_gripper_state())
        prev_dist = 10000
        cnt = 0
        while True:
            print("close: {}, target: {}\n".format(self.get_gripper_state(), target))
            self._set_gripper_velocity(gripper_vel)
            if render:
                img = self.physics.render(camera_id='static_camera')
                cv_render(img)
            with ignored_physics_warning():
                self.physics.step()
            cur_dist = self.get_gripper_state() - target
            if self.get_gripper_state() < target:
                break
            prev_dist = cur_dist
            cnt += 1

    def _pick_and_place_one_step(self, arm_st_loc, arm_en_loc, velocity=None,
                                 grip_vel=0.1, grip_closed=0.025, init_height=1.0, grip_force=5):
        if velocity is None:
            velocity = self.arm_move_velocity
        if self.prev_action_finished:
            self.maxblockheight = 0
            ropePos = self.get_rope_offset_location(3)
            hovered = [ropePos[0], ropePos[1]-0.015, init_height]
            self.set_arm_location(hovered)
            self.set_gripper_state(0.035)
            move_dist = self.get_point_dist(arm_en_loc, arm_st_loc)
            if move_dist == 0:
                self.prev_action_finished = True
                return
            self.arm_stage = 0
            self.move_arm_prev_dist = 10000
            self.prev_action_finished = False
        # Move down
        if self.arm_stage == 0:
            self.move_arm_move_dir = np.asarray([0.0, 0.0, -2.0])
            self.set_gripper_state(0.03)
            self._set_arm_velocity(velocity * self.move_arm_move_dir)
            with ignored_physics_warning():
                self.physics.step()
            #cur_dist = self._get_endpoints_distance(self.get_arm_location(), arm_st_loc)
            if self.get_arm_location()[2]-0.006 <= arm_st_loc[2]:
                self.arm_stage = 1
            print("grip block loc: {}".format(self.get_grip_block_location()))
            #self.move_arm_prev_dist = cur_dist
        # Close gripper
        elif self.arm_stage == 1:
            self._set_arm_velocity(0)
            self._set_gripper_force(-grip_force)

            #self.set_grip_block_force(0.001)
            #self._set_gripper_velocity(-grip_vel)
            #if self.get_gripper_state() < grip_closed:
            #    self._set_gripper_velocity(0)
            self.move_arm_move_dir = np.asarray([0.0, 0.0, 0.0])
            #self.set_grip_block_velocity(velocity * self.move_arm_move_dir)
            with ignored_physics_warning():
                self.physics.step()
            cur_dist = self.get_gripper_state() - grip_closed
            if cur_dist < 0:
                finger_locs = self.get_finger_location()
                target_seg = self.get_closest_rope_point(finger_locs[0], finger_locs[1])
                if target_seg is not None:
                    i, point = target_seg
                    self.weld_rope_offset(i)
                self.arm_stage = 2
                """
                i, point = self.get_closest_rope_point(finger_locs[0], finger_locs[1])
                self.weld_rope_offset(i)
                self.arm_stage = 2
                """

        # Move up
        elif self.arm_stage == 2:
            self.move_arm_move_dir = np.asarray([0.0, 0.0, 0.2])
            self._set_gripper_force(0)
            self.set_gripper_state(0.02)
            ropePos = self.get_arm_location()
            new_rope_frc = self.get_arm_xfrc()
            #self.set_rope_offset_force_partial(3, 2, 0.2)
            ropePos[2] = ropePos[2] -0.18

            #self.set_rope_offset_location(3, ropePos)
            #self._set_gripper_velocity(-grip_vel)
            #if self.get_gripper_state() < grip_closed:
            #    self._set_gripper_velocity(0)
            self._set_arm_velocity(self.move_arm_move_dir)
            with ignored_physics_warning():
                self.physics.step()
            # cur_dist = self._get_endpoints_distance(self.get_arm_location(), arm_st_loc)
            if self.get_arm_location()[2] >= init_height:
                self.arm_stage = 3
                # self.move_arm_prev_dist = cur_dist
        # Move across
        elif self.arm_stage == 3:
            move_dist = self.get_point_dist(arm_en_loc, self.get_arm_location())
            self._set_gripper_force(0)
            self.set_gripper_state(0.02)
            blockPos = self.get_arm_location()
            blockPos[2] = blockPos[2] - 0.18

            #self.set_rope_start_location(blockPos)
            #self._set_gripper_velocity(-grip_vel)
            #if self.get_gripper_state() < grip_closed:
            #    self._set_gripper_velocity(0)
            if move_dist == 0:
                self.prev_action_finished = True
                return
            self.move_arm_move_dir = (arm_en_loc - self.get_arm_location()) / move_dist
            self.move_arm_move_dir[2] = 0
            self._set_arm_velocity(self.arm_move_velocity*self.move_arm_move_dir)
            with ignored_physics_warning():
                self.physics.step()
            cur_dist = self._get_endpoints_distance(self.get_arm_location()[0:2], arm_en_loc[0:2])
            if cur_dist > self.move_arm_prev_dist:
                self.arm_stage = 4
                self.unweld_rope_offset()
            self.move_arm_prev_dist = cur_dist
        else:
            self._set_gripper_velocity(grip_vel)
            self.move_arm_move_dir = np.asarray([0.0, 0.0, 0.0])
            self._set_arm_velocity(velocity * self.move_arm_move_dir)
            with ignored_physics_warning():
                self.physics.step()
            cur_dist = self.get_gripper_state()
            blockPos = self.get_rope_offset_location(3)
            if blockPos[2] < 0.72:
                print("finished action")
                self.prev_action_finished = True
        current_block_pos = self.get_block_location()

        self.maxblockheight = max(self.maxblockheight,current_block_pos[2])
        print("arm state: {}, block forces: {}, block state: {}".format(self.arm_stage,
                                                                              self.physics.data.qfrc_unc[self.qfrc_block_unc],
                                                                              self.get_block_location()))



    def _move_arm_by_endpoints_one_step(self, arm_st_loc, arm_en_loc, velocity=None):
        # Move the arm from the start loc to the end loc for only one step
        if velocity is None:
            velocity = self.arm_move_velocity
        if self.prev_action_finished:
            self.set_arm_location(arm_st_loc)
            move_dist = self.get_point_dist(arm_en_loc, arm_st_loc)
            if move_dist == 0:
                self.prev_action_finished = True
                return
            self.move_arm_move_dir = (arm_en_loc - arm_st_loc) / move_dist

            self.move_arm_prev_dist = 10000
            self.prev_action_finished = False
        self._set_arm_velocity(self.move_arm_move_dir * velocity)
        with ignored_physics_warning():
            self.physics.step()
        cur_dist = self._get_endpoints_distance(self.get_arm_location(), arm_en_loc)
        if cur_dist > self.move_arm_prev_dist:
            self.prev_action_finished = True
        self.move_arm_prev_dist = cur_dist

    def get_target_goal_state(self):
        # thetas = self.physics.data.qpos[self.state_target_rope_inds[7:]]
        # return np.hstack([np.cos(thetas), np.sin(thetas)])
        return self.physics.data.xpos[self.xpos_target_rope_inds, :2].flatten().copy()

    def get_achieved_goal_state(self):
        # ref_pose = self.physics.data.qpos[self.state_rope_ref_inds[-4:]]
        # thetas = self.physics.data.qpos[self.state_rope_inds[7:]]
        # return np.hstack([ref_pose, np.cos(thetas), np.sin(thetas)])
        # Only need to achieve the angles
        # return np.hstack([np.cos(thetas), np.sin(thetas)])
        return self.physics.data.xpos[self.xpos_rope_inds, :2].flatten().copy()

    @staticmethod
    def _get_endpoints_distance(pt1, pt2):
        return np.linalg.norm(pt1 - pt2)

    @property
    def EnvWrapper(self):
        if self.action_type == 'endpoints':
            return EndpointsPushEnvViewerWrapper
        elif self.action_type == 'pick_and_place':
            return EndpointsPickAndPlaceEnvViewerWrapper
        else:
            return None

    @property
    def PolicyWrapper(self):
        if self.action_type == 'endpoints':
            return EndpointsPushPolicyViewerWrapper
        elif self.action_type == 'pick_and_place':
            return EndpointsPickAndPlacePolicyViewerWrapper
        else:
            return None


# These wrappers takes in the original environment and policy where the policy is parameterized by endpoints of pushing
# and transform them into the ones that takes each step as velocity control in cartisan space

class EndpointsPushEnvViewerWrapper(object):
    def __init__(self, env):
        self.env = env
        self.physics = self.env.physics
        assert self.env.action_type == 'endpoints'
        self.env.action_type = 'endpoints_visualization'
        self.prev_action_finished = self.env.prev_action_finished
        self.time_count = 0

    def reset(self):
        self.time_count = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.prev_action_finished = self.env.prev_action_finished
        if self.prev_action_finished:
            self.time_count += 1
        info['time_count'] = self.time_count
        return obs, reward, done, info


class EndpointsPushPolicyViewerWrapper(object):
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
        self.env.prev_action_finished = True
        self.last_action = None

    def get_actions(self, *args, **kwargs):
        if self.env.prev_action_finished:
            action = self.policy.get_actions(*args, **kwargs)
            print(action)
        else:
            action = self.last_action
        assert len(action) == 4
        self.last_action = action
        return action

class EndpointsPickAndPlaceEnvViewerWrapper(object):
    def __init__(self, env):
        self.env = env
        self.physics = self.env.physics
        assert self.env.action_type == 'pick_and_place'
        self.env.action_type = 'pick_and_place_visualization'
        self.env.arm_stage = 0
        self.prev_action_finished = self.env.prev_action_finished
        self.time_count = 0

    def reset(self):
        self.time_count = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.prev_action_finished = self.env.prev_action_finished
        if self.prev_action_finished:
            self.time_count += 1
        info['time_count'] = self.time_count
        return obs, reward, done, info

class EndpointsPickAndPlacePolicyViewerWrapper(object):
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
        self.env.prev_action_finished = True
        self.last_action = None

    def get_actions(self, *args, **kwargs):
        if self.env.prev_action_finished:
            action = self.policy.get_actions(*args, **kwargs)
            print(action)
        else:
            action = self.last_action
        assert len(action) == 4
        self.last_action = action
        return action

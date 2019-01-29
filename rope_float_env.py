# Created by Xingyu Lin, 2019/1/8                                                                                  
from dfm_env.sawyer_float_env import SawyerFloatEnv
import numpy as np
from .utils.util import get_name_arr_and_len, ignored_physics_warning, cv_render
import os.path

# def generate_color_map():
#     from random import shuffle
#     import matplotlib.pyplot as plt
#     # Set the rope to be colorful
#     cmap = plt.get_cmap('tab20')
#     cmap_rgba = [cmap(x / 20.) for x in range(20)]
#     shuffle(cmap_rgba)

default_color_map = np.array([[0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0],
                              [0.7725490196078432, 0.6901960784313725, 0.8352941176470589, 1.0],
                              [0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0],
                              [1.0, 0.4980392156862745, 0.054901960784313725, 1.0],
                              [0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0],
                              [0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0],
                              [0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1.0],
                              [1.0, 0.596078431372549, 0.5882352941176471, 1.0],
                              [0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0],
                              [0.596078431372549, 0.8745098039215686, 0.5411764705882353, 1.0],
                              [0.7686274509803922, 0.611764705882353, 0.5803921568627451, 1.0],
                              [0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0],
                              [1.0, 0.7333333333333333, 0.47058823529411764, 1.0],
                              [0.7803921568627451, 0.7803921568627451, 0.7803921568627451, 1.0],
                              [0.6196078431372549, 0.8549019607843137, 0.8980392156862745, 1.0],
                              [0.9686274509803922, 0.7137254901960784, 0.8235294117647058, 1.0],
                              [0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0],
                              [0.8588235294117647, 0.8588235294117647, 0.5529411764705883, 1.0],
                              [0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0],
                              [0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0],
                              ])


class RopeFloatEnv(SawyerFloatEnv):
    def __init__(self, distance_threshold=5e-2, goal_push_num=2, visualization_mode=False, **kwargs):
        model_path = 'tasks/rope_float.xml'
        self.goal_push_num = goal_push_num
        cached_file = './dfm_env/cached/generated_rope_{}.npz'.format(goal_push_num)
        if os.path.exists(cached_file):
            self.use_cached_inits_goals = True
            data = np.load(cached_file)
            self.all_init_qpos = data['all_init_qpos']
            self.all_target_qpos = data['all_target_qpos']
            print('Rope_float_env: using cached init poses and goal poses with {} poses'.format(len(data['all_init_qpos'])))
            # TODO check why this is being executed multiple times
            # import traceback
            # traceback.print_stack()
        else:
            self.use_cached_inits_goals = False

        self.visualization_mode = visualization_mode
        super().__init__(model_path=model_path, distance_threshold=distance_threshold, **kwargs)
        if not self.visualization_mode:
            # Hide the arm in case we are not visualizing
            self.physics.model.geom_rgba[self.geom_rgba_arm_inds, 3] = 0
            # Also hide the target rope if image observation is used
            if self.use_visual_observation:
                self.physics.model.geom_rgba[self.geom_rgba_target_rope_inds, 3] = 0.

        num_link = len(self.geom_rgba_rope_inds)
        self.physics.model.geom_rgba[self.geom_rgba_rope_inds, :3] = default_color_map[:num_link, :3] * 0.6
        self.physics.model.geom_rgba[self.geom_rgba_target_rope_inds, :3] = default_color_map[:num_link, :3] * 0.6

    # Rope specific helper functions
    def _sample_rope_init_xpos(self):
        # Sample one of the rope elements and put the gripper on top of it
        sampled_idx = np.random.choice(self.xpos_rope_inds, 1)
        return self.physics.data.xpos[sampled_idx] + np.array([0, 0, 0.20])

    def _reset_sim(self, restore_info = None):
        # Sample goal and render image, Get the goal after the environment is stable
        # print('camera:', self.get_camera_info(-1))
        with self.physics.reset_context():
            self._reset_all_to_init_pos()
            if restore_info is None:
                if not self.use_cached_inits_goals:
                    self._random_push_rope(push_num=1)  # Init rope distribution
                    one_push_rope_qpos = self.physics.data.qpos[self.qpos_rope_inds]
                    # Set the target rope qpos and reset the rope qpos
                    self._random_push_rope(push_num=None)  # Push k times according to the environment
                    target_rope_qpos = self.physics.data.qpos[self.qpos_rope_inds].copy()
                else:
                    cached_idx = np.random.randint(0, len(self.all_init_qpos))
                    one_push_rope_qpos = self.all_init_qpos[cached_idx]
                    target_rope_qpos = self.all_target_qpos[cached_idx]
            else:
                one_push_rope_qpos = restore_info['one_push_rope_qpos']
                target_rope_qpos = restore_info['target_rope_qpos']
            self._one_push_rope_qpos = one_push_rope_qpos.copy()
            self._target_rope_qpos = target_rope_qpos.copy()

            self.physics.data.qpos[self.qpos_rope_inds] = one_push_rope_qpos
            self.physics.data.qvel[self.qpos_rope_inds] = np.zeros(len(self.qpos_rope_inds))
            self.physics.data.qpos[self.qpos_target_rope_inds] = target_rope_qpos
            self.physics.forward()
            # self.set_arm_location(self.init_arm_xpos)

            if self.use_image_goal or True:
                # self.physics.data.qpos[self.qpos_target_rope_ref_inds[1]] += self.visualization_offset
                target_original_transparancy = self.physics.model.geom_rgba[self.geom_rgba_target_rope_inds, 3][0]
                self.physics.model.geom_rgba[self.geom_rgba_target_rope_inds, 3] = 1.
                self.physics.model.geom_rgba[self.geom_rgba_rope_inds, 3] = 0
                self.physics.forward()
                # self.physics.model.geom_rgba[1, :] = np.asarray([0., 0., 0, 0.])  # Make the goal transparent
                self.goal_observation = self.render(depth=False)
                # self.physics.data.qpos[self.qpos_target_rope_ref_inds[1]] -= self.visualization_offset
                self.physics.model.geom_rgba[self.geom_rgba_target_rope_inds, 3] = target_original_transparancy
                self.physics.model.geom_rgba[self.geom_rgba_rope_inds, 3] = 1.
                # Set the target qpos
                # self.physics.data.qpos[self.state_target_rope_inds] = self.physics.data.qpos[self.state_rope_inds]
                # self.physics.data.qvel[self.state_target_rope_inds] = 0
        self.goal_state = self.get_target_goal_state()
        return True

    def _sample_rope_neighbourhood(self):
        # Sample one of the rope elements and put the gripper on top of it
        sampled_idx = np.random.choice(self.xpos_rope_inds, 1)
        point_start = self.physics.data.xpos[sampled_idx][0].copy()
        point_end = point_start.copy()
        delta_x = (np.random.random() - 0.5) / 5
        delta_y = np.sign(np.random.random() - 0.5) * 0.2
        delta_xy = [delta_x, delta_y]
        point_start[:2] += delta_xy
        point_end[:2] -= delta_xy
        # Only need this if the sampled neighbourhood is for the sawyer gripper
        # point_start[2] = self.boundary_range[2][0]  # Account for the height of the gripper
        # point_end[2] = self.boundary_range[2][0]  # Account for the height of the gripper
        return point_start, point_end

    def _random_push_rope(self, push_num=None):
        # Sample two points near the rope and let the block to move from one point to another point
        # Keep trying until the position of the rope changes

        # init_rope_state = self.get_achieved_goal_state()
        if push_num is None:
            push_num = np.random.randint(0, self.goal_push_num) + 1

        for _ in range(push_num):
            start_pt, end_pt = self._sample_rope_neighbourhood()
            # Fix the height of the arm when moving
            start_pt[2] = end_pt[2] = self.arm_height
            self._move_arm_by_endpoints(start_pt, end_pt, render=False)
        return

    def get_restore_info(self):
        return {
            'one_push_rope_qpos': self._one_push_rope_qpos,
            'target_rope_qpos': self._target_rope_qpos
        }

# Created by Xingyu Lin, 2018/11/18                                                                                  
from rope_env import RopeEnv
from dm_control import viewer
import numpy as np
import cv2

def random_policy(time_step=None):
    del time_step  # Unused.
    # print(env.action_spec().minimum,env.action_spec().maximum,env.action_spec().shape)
    lo = -1000 * np.ones((8, 1))
    hig = 1000 * np.ones((8, 1))
    return np.random.uniform(low=lo,
                             high=hig,
                             size=(8, 1))


if __name__ == '__main__':

    env = RopeEnv(use_visual_observation=False, use_image_goal=False)
    # viewer.launch(env)

    # env.reset()
    # print(env.physics.named.model.body_pos)
    print(env.goal_state)
    print(env.physics.get_state())
    exit()
    # action = random_policy().squeeze()
    action = np.ones((8,)) * 100
    while True:

        # print(env.physics.data.ctrl)
        pixels, _, _, _ = env.step(action)
        pixels = pixels['observation']
        pixels = pixels / 255.0
        pixels = pixels[:, :, ::-1]  # BGR to RGB
        print(action)
        cv2.imshow('display', pixels)
        cv2.waitKey(10)

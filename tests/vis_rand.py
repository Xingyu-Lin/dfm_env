# Created by Xingyu Lin, 2018/11/18                                                                                  
from rope_env import RopeEnv
import numpy as np
import cv2

def random_policy(time_step=None):
    del time_step  # Unused.
    # print(env.action_spec().minimum,env.action_spec().maximum,env.action_spec().shape)
    lo = -1 * np.ones((8, 1))
    hig = 1 * np.ones((8, 1))
    return np.random.uniform(low=lo,
                             high=hig,
                             size=(8, 1))


if __name__ == '__main__':

    env = RopeEnv()

    while True:
        action = random_policy().squeeze()
        # print(env.physics.data.ctrl)
        pixels, _, _, _ = env.step(action)
        print(action)
        pixels = pixels['observation']
        pixels = pixels / 255.0
        pixels = pixels[:, :, ::-1]  # BGR to RGB
        cv2.imshow('display', pixels)
        cv2.waitKey(10)

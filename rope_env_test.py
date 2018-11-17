from rope_env import RopeEnv
import cv2
import numpy as np

def random_policy(time_step=None):
  del time_step  # Unused.
  #print(env.action_spec().minimum,env.action_spec().maximum,env.action_spec().shape)
  lo = np.zeros((8,1))
  hig = np.zeros((8,1))
  return np.random.uniform(low=lo,
                           high=hig,
                           size=(8,1))

if __name__ == '__main__':
    env = RopeEnv()

    while True:
        action = random_policy().squeeze()
        # print(env.physics.data.ctrl)
        pixels, _, _, _ = env.step(action)
        print(pixels)
        pixels = pixels['observation']
        pixels = pixels / 255.0
        pixels = pixels[:, :, ::-1]  # BGR to RGB
        cv2.imshow('display', pixels)
        cv2.waitKey(100)
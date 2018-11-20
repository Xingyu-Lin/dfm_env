# Created by Xingyu Lin, 2018/11/18
from dm_control.rl.environment import TimeStep, StepType
from dm_control import mujoco


class ViewerWrapper(object):
    def __init__(self, env):
        self.env = env
        self.physics = self.env.physics

    def reset(self):
        obs = self.env.reset()
        return TimeStep(StepType.FIRST, None, None, obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return TimeStep(StepType.MID, reward, 1.0, obs)

    def action_spec(self):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        return mujoco.action_spec(self.env.physics)

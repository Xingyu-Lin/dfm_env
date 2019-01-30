# Created by Xingyu Lin, 2018/11/18
from dm_control.rl.environment import TimeStep, StepType
from dm_control import mujoco


class ViewerWrapper(object):
    def __init__(self, env, eval_params={'T': 100000}, saved_episodes=None):
        self.env = env
        self.physics = self.env.physics
        self.T = eval_params['T']
        self.time_count = 0
        self.saved_episodes = saved_episodes
        self.replay_cnt = -1

    def reset(self):
        self.replay_cnt += 1
        if self.saved_episodes is not None:
            obs = self.env.reset(restore_info=self.saved_episodes[self.replay_cnt]['restore_info'])
        else:
            obs = self.env.reset()
        self.time_count = 0
        return TimeStep(StepType.FIRST, None, None, obs)

    def step(self, action):
        if self.saved_episodes is not None:
            action = self.saved_episodes[self.replay_cnt]['u'][self.time_count]
        obs, reward, done, info = self.env.step(action)
        if 'time_count' in info:
            self.time_count = info['time_count']
        else:
            self.time_count += 1
        if self.time_count == self.T:
            print('reward: ', reward)
            return TimeStep(StepType.LAST, reward, 1.0, obs)
        else:
            return TimeStep(StepType.MID, reward, 1.0, obs)

    def action_spec(self):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        return mujoco.action_spec(self.env.physics)


def ViewerPolicyWrapper(ddpg_policy, eval_params):
    '''
    policy: An optional callable corresponding to a policy to execute
        within the environment. It should accept a `TimeStep` and return
        a numpy array of actions conforming to the output of
        `environment.action_spec()`.
    '''

    def dm_policy(time_step=None):
        obs = time_step[3]
        action = ddpg_policy.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'],
                                         compute_Q=False,
                                         noise_eps=eval_params['noise_eps'] if not eval_params['exploit'] else 0.,
                                         random_eps=eval_params['random_eps'] if not eval_params['exploit'] else 0.,
                                         use_target_net=eval_params['use_target_net'])
        # print('action:', action)
        return action

    return dm_policy

# Created by Xingyu Lin, 2018/11/12
# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Pendulum domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco, viewer
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

_DEFAULT_TIME_LIMIT = 500

from dm_control.utils import io as resources
import os


def read_model(model_filename):
    """Reads a model XML file and returns its contents as a string."""
    return resources.GetResource(model_filename)


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return read_model('rope.xml'), common.ASSETS


from dm_control.mujoco import Physics


def rope_env(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
    """Returns rope manipulation task."""

    physics = Physics.from_xml_string(*get_model_and_assets())
    task = SwingUp(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


# class Physics(mujoco.Physics):
#     """Physics simulation with additional features for the Pendulum domain."""
#
#     def pole_vertical(self):
#         """Returns vertical (z) component of pole frame."""
#         return self.named.data.xmat['pole', 'zz']
#
#     def angular_velocity(self):
#         """Returns the angular velocity of the pole."""
#         return self.named.data.qvel['hinge'].copy()
#
#     def pole_orientation(self):
#         """Returns both horizontal and vertical components of pole frame."""
#         return self.named.data.xmat['pole', ['zz', 'xz']]
#

class SwingUp(base.Task):
    """A Pendulum `Task` to swing up and balance the pole."""

    def __init__(self, random=None):
        """Initialize an instance of `Pendulum`.

        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super(SwingUp, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Pole is set to a random angle between [-pi, pi).

        Args:
          physics: An instance of `Physics`.

        """
        return
        physics.named.data.qpos['hinge'] = self.random.uniform(-np.pi, np.pi)

    def get_observation(self, physics):
        """Returns an observation.

        Observations are states concatenating pole orientation and angular velocity
        and pixels from fixed camera.

        Args:
          physics: An instance of `physics`, Pendulum physics.

        Returns:
          A `dict` of observation.
        """
        return None
        obs = collections.OrderedDict()
        obs['orientation'] = physics.pole_orientation()
        obs['velocity'] = physics.angular_velocity()
        return obs

    def get_reward(self, physics):
        return 0
        return rewards.tolerance(physics.pole_vertical(), (_COSINE_BOUND, 1))


if __name__ == '__main__':
    env = rope_env()
    env.physics.named.data.qpos['arm_j6'] = 0.5
    viewer.launch(env)
# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of gym wrappers."""

import importlib
import os
from collections import namedtuple
from os.path import join as pjoin

import gym
import yaml
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from value_dice.wrappers.absorbing_wrapper import AbsorbingWrapper
from value_dice.wrappers.normalize_action_wrapper import NormalizeBoxActionWrapper
from value_dice.wrappers.normalize_action_wrapper import check_and_normalize_box_actions
from value_dice.wrappers.normalize_state_wrapper import NormalizeStateWrapper

try:
    import sb3_contrib
except ImportError:
    raise ImportError("Cannot import sb3_contrib")


def get_wrapper_class(hyperparams):
    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if "env_wrapper" in hyperparams:
        wrapper_name = hyperparams.get("env_wrapper")

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        for wrapper_name in wrapper_names:
            if isinstance(wrapper_name, dict):
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env):
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None


def create_zoo_env(env_id, stats_dir, seed, hyperparams, should_render=False):
    env_wrapper = get_wrapper_class(hyperparams)

    vec_env_cls = DummyVecEnv
    if "Bullet" in env_id and should_render:
        vec_env_cls = SubprocVecEnv

    env = make_vec_env(
        env_id,
        seed=seed,
        wrapper_class=env_wrapper,
        vec_env_cls=vec_env_cls
    )

    if stats_dir is not None:
        if hyperparams["normalize"]:
            norm_fpath = pjoin(stats_dir, "vecnormalize.pkl")

            if os.path.exists(norm_fpath):
                env = VecNormalize.load(norm_fpath, env)
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {norm_fpath} not found")

    max_episode_steps = gym.make(env_id).spec.max_episode_steps
    Spec = namedtuple("Spec", ["max_episode_steps"])
    env.spec = Spec(max_episode_steps=max_episode_steps)

    return env


def load_saved_hyperparams(stats_path, norm_reward=False):
    config_fpath = pjoin(stats_path, "config.yml")

    with open(config_fpath, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)
    hyperparams["normalize"] = hyperparams.get("normalize", False)

    if hyperparams["normalize"]:
        normalize_kwargs = {"norm_obs": hyperparams["normalize"], "norm_reward": norm_reward}
        hyperparams["normalize_kwargs"] = normalize_kwargs

    return hyperparams


def create_il_env(env_name, seed, shift, scale):
    """Create a gym environment for imitation learning.

    Args:
      env_name: an environment name.
      seed: a random seed.
      shift: a numpy vector to shift observations.
      scale: a numpy vector to scale observations.

    Returns:
      An initialized gym environment.
    """
    # env = gym.make(env_name)
    expert_dir = pjoin("/Users/fedepiro/Projects/topographic-nn/environments", "experts", env_name)
    stats_dir = pjoin(expert_dir, env_name)
    hyperparams = load_saved_hyperparams(stats_dir)
    env = create_zoo_env(env_name, stats_dir, seed, hyperparams)
    env.reward_range = (-float('inf'), float('inf'))
    env = check_and_normalize_box_actions(env)
    env.seed(seed)

    if shift is not None:
        env = NormalizeStateWrapper(env, shift=shift, scale=scale)

    return AbsorbingWrapper(env)

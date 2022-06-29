import logging
from typing import NamedTuple

import gym
import numpy as np

from rlberry.envs.utils import process_env
from rlberry.utils.jit_setup import numba_jit


logger = logging.getLogger(__name__)

# Environment processing (for vectorized environments)
def process_ppo_env(env, seeder, num_envs=1):
    def make(env, seeder):
        env = process_env(env, seeder, copy_env=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # important for logging
        return env

    if num_envs == 1:
        return make(env, seeder)
    seeders = seeder.spawn(num_envs)
    return gym.vector.SyncVectorEnv([make(env, seeder) for seeder in seeders])


# Compute targets
@numba_jit
def lambda_returns(r_t, discount_t, v_tp1, lambda_):
    """
    Computer lambda returns

    Parameters
    ----------
    r_t: array
        Array of shape (batch_dim, time_dim) containing the rewards.
    discount_t: array
        Array of shape (batch_dim, time_dim) containing the discounts (0.0 if terminal state).
    v_tp1: array
        Array of shape (batch_dim, time_dim) containing the values at timestep t+1
    lambda_ : float in [0, 1]
        Lambda-returns parameter.
    """
    returns = np.zeros_like(r_t)
    aux = v_tp1[:, -1]
    time_dim = v_tp1.shape[1]
    for tt in range(time_dim):
        i = time_dim - tt - 1
        aux = r_t[:, i] + discount_t[:, i] * (
            (1 - lambda_) * v_tp1[:, i] + lambda_ * aux
        )
        returns[:, i] = aux
    return returns


# Data buffer
class Batch(NamedTuple):
    data: dict
    info: dict


class RolloutBuffer:
    """
    Rollout buffer that allows sampling data with shape (batch_size,
    num_trajectories, ...).

    Parameters
    ----------
    max_replay_size: int
        Maximum number of transitions that can be stored
    rng: numpy.random.Generator
        Numpy random number generator.
        See https://numpy.org/doc/stable/reference/random/generator.html
    max_episode_steps: int, optional
        Maximum length of an episode
    """

    def __init__(self, rng, num_rollout_steps, num_envs=1):
        self._rng = rng
        self._num_rollout_steps = num_rollout_steps
        self._num_envs = num_envs
        self._tags = []
        self._data = dict()
        self._dtypes = dict()

    @property
    def data(self):
        """Dict containing all stored data."""
        return self._data

    @property
    def tags(self):
        """Tags identifying the entries in the replay buffer."""
        return self._tags

    @property
    def dtypes(self):
        """Dict containing the data types for each tag."""
        return self._dtypes

    @property
    def num_rollout_steps(self):
        """Number of steps to take in each environment per policy rollout."""
        return self._num_rollout_steps

    @property
    def num_envs(self):
        return self._num_envs

    def __len__(self):
        return len(self._data[self.tags[0]])

    def full(self):
        """Returns True if the buffer is full."""
        return len(self) == self.num_rollout_steps

    def clear(self):
        """Clear data in replay."""
        for tag in self._data:
            self._data[tag] = []

    def setup_entry(self, tag, dtype):
        """Configure replay buffer to store data.
        Parameters
        ----------
        tag : str
            Tag that identifies the entry (e.g "observation", "reward")
        dtype : obj
            Data type of the entry (e.g. `np.float32`). Type is not
            checked in :meth:`append`, but it is used to construct the numpy
            arrays returned by the :meth:`sample`method.
        """
        assert len(self) == 0, "Cannot setup entry on non-empty buffer."
        if tag in self._data:
            raise ValueError(f"Entry {tag} already added to replay buffer.")
        self._tags.append(tag)
        self._dtypes[tag] = dtype
        self._data[tag] = []

    def append(self, data):
        """Store data.
        Parameters
        ----------
        data : dict
            Dictionary containing scalar values, whose keys must be in self.tags.
        """
        # Append data
        assert set(data.keys()) == set(self.tags), "Data keys must be in self.tags"
        assert len(self) < self.num_rollout_steps, "Buffer is full."
        for tag in self.tags:
            self._data[tag].append(data[tag])

    def get(self, squeeze=False):
        """Returns the collected data.
        Data have shape (T, E, ...), where
        T = num_rollout_steps
        E = number of environments
        and represents a batch of trajectories.

        Parameters
        ----------
        squeeze : bool, optional
            If True, the data is squeezed to shape (T, ...) when using a
            single environment.

        Returns
        -------
        Returns a NamedTuple :code:`batch` where:
        * :code:`batch.data` is a dict such that `batch.data[tag]` is a numpy array
        containing data stored for a given tag.
        """
        batch_data = dict()

        for tag in self.tags:
            batch_data[tag] = np.array(self._data[tag], dtype=self._dtypes[tag])

            # Reshape to (T, E, ...) if necessary
            if self.num_envs == 1 and not squeeze:
                batch_data[tag] = np.expand_dims(batch_data[tag], axis=1)

        batch = Batch(data=batch_data, info=dict())
        return batch

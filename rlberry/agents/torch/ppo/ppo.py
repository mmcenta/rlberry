import logging
import inspect
import time
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch.distributions import Categorical
from gym import spaces

from rlberry import types
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.torch.utils.training import (
    loss_function_factory,
    model_factory,
    optimizer_factory,
    size_model_config,
)
from rlberry.agents.torch.ppo.ppo_utils import (
    process_ppo_env,
    lambda_returns,
    RolloutBuffer,
)
from rlberry.agents.torch.dqn.dqn_utils import polynomial_schedule, lambda_returns
from rlberry.utils.torch import choose_device
from rlberry.utils.factory import load


logger = logging.getLogger(__name__)


def default_policy_net_fn(env, **kwargs):
    """
    Returns a default policy network.
    """
    del kwargs
    model_config = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": (64, 64),
        "reshape": False,
    }
    model_config = size_model_config(env, **model_config)
    return model_factory(**model_config)


def default_value_net_fn(env, **kwargs):
    """
    Returns a default valu network.
    """
    del kwargs
    model_config = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": (64, 64),
        "out_size": 1,
        "reshape": False,
    }
    model_config = size_model_config(env, **model_config)
    return model_factory(**model_config)


class PPOAgent(AgentWithSimplePolicy):
    """PPO Agent based on PyTorch.

    Notes
    -----
    Uses Generalized Advantage Estimation (GAE) for computing targets by default. To recover
    the standard PPO, set :code:`use_gae = False`.

    Parameters
    ----------
    env: :class:`~rlberry.types.Env`
        Environment, can be a tuple (constructor, kwargs)
    gamma: float, default = 0.99
        Discount factor.
    minibatch_size: int, default = 32
        Size of minibatches used in updates.
    num_steps: int, default = 128
        Number of steps to take in each environment per policy rollout.
    num_epochs: int, default = 4
        Number of update epochs per rollout.
    clip_coef: float, default = 0.2
        The surrogate clipping coeficient.
    clip_vloss: bool, default = True
        Toggles whether or not to use a clipped loss for the value function, as per the paper.
    learning_rate: float, default = 2.5e-4
        Optimizer learning rate.
    optimizer_type: {"ADAM", "RMS_PROP"}
        Optimization algorithm.
    value_loss_fn: {"l1", "l2", "smooth_l1"}, default: "l2"
        Loss function used to compute Bellman error of the value function.
    normalize_advantage: bool, default = True
        Whether or not to normalize the advantages when computing targets.
    use_gae: bool, default = True
        Whether or not to use Generalized Advantage Estimation.
    gae_lambda: float, default = 0.95
        Lambda parameter for GAE.
    ent_coef: float, default = 0.01
        Coefficient of the entropy regularization term.
    vf_coef: float, default = 0.5
        Coefficient of the value function loss.
    max_grad_norm: float, default = 0.5
        Maximum norm for gradient clipping.
    device: str
        Torch device, see :func:`~rlberry.utils.torch.choose_device`.
    policy_net_fn: Callable, str or None
        Function/constructor that returns a torch module for the policy network:
        :code:`policy = policy_net_fn(env, **kwargs)`.

        Policy module requirements:
        * Input shape = (batch_dim, obs_dims)
        * Ouput shape = (batch_dim, number_of_actions)
    policy_net_kwargs: dict
        Keyword arguments for the policy network constructor.
    value_net_fn: Callable, str or None
        Function/constructor that returns a torch module for the value network:
        :code:`value = value_net_fn(env, **kwargs)`.

        Value module requirements:
        * Input shape = (batch_dim, obs_dims)
        * Ouput shape = (batch_dim, 1)
    value_net_kwargs: dict
        Keyword arguments for the value network constructor.
    eval_interval : int, default = None
        Interval (in number of transitions) between agent evaluations in fit().
        If None, never evaluate.

    References
    ----------
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    "Proximal policy optimization algorithms."
    In arXiv preprint arXiv:1707.06347.
    """

    name = "PPO"

    def __init__(
        self,
        env: types.Env,
        gamma: float = 0.99,
        num_steps: int = 128,
        num_envs: int = 1,
        num_epochs: int = 4,
        clip_coef: float = 0.2,
        clip_vloss: bool = False,
        minibatch_size: int = 32,
        learning_rate: float = 2.5e-4,
        optimizer_type: str = "ADAM",
        value_loss_fn: str = "l2",
        normalize_advantage: bool = True,
        use_gae: bool = True,
        gae_lambda: int = 0.95,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda:best",
        policy_net_fn: Optional[Callable[..., torch.nn.Module]] = None,
        policy_net_kwargs: Optional[dict] = None,
        value_net_fn: Optional[Callable[..., torch.nn.Module]] = None,
        value_net_kwargs: Optional[dict] = None,
        # lr_annealing_fn: Optional[Callable[..., torch.nn.Module]] = None, TODO: add this
        target_kl: Optional[float] = None,
        eval_interval: Optional[int] = None,
        num_eval_episodes: int = 10,
        **kwargs,
    ):
        # For all parameters, define self.param = param
        _, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        env = self.env
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Discrete)

        # Create vectorized environment if needed
        self.num_envs = num_envs
        self.env = process_ppo_env(env, self.seeder, num_envs=num_envs)

        # Policy network, torch device
        self._device = choose_device(device)
        if isinstance(policy_net_fn, str):
            policy_net_ctor = load(policy_net_fn)
        elif policy_net_fn is None:
            policy_net_ctor = default_policy_net_fn
        else:
            policy_net_ctor = policy_net_fn
        policy_net_kwargs = policy_net_kwargs or dict()
        self._policy_net = policy_net_ctor(env, **policy_net_kwargs)

        # Value network, torch device
        if isinstance(value_net_fn, str):
            value_net_ctor = load(value_net_fn)
        elif value_net_fn is None:
            value_net_ctor = default_value_net_fn
        else:
            value_net_ctor = value_net_fn
        value_net_kwargs = value_net_kwargs or dict()
        self._value_net = value_net_ctor(env, **value_net_kwargs)

        # Optimizer and loss
        optimizer_kwargs = {"optimizer_type": optimizer_type, "lr": learning_rate}
        self._optimizer = optimizer_factory(self.parameters, **optimizer_kwargs)
        self._loss_function = loss_function_factory(value_loss_fn, reduction="none")

        # Training params
        self._eval_interval = eval_interval
        self._eval_episodes = num_eval_episodes

        # Setup rollout buffer
        if hasattr(self.env, "_max_episode_steps"):
            max_episode_steps = self.env._max_episode_steps
        else:
            max_episode_steps = np.inf
        self._max_episode_steps = max_episode_steps

        self._buffer = RolloutBuffer(
            rng=self.rng,
            num_rollout_steps=num_steps,
            num_envs=self.num_envs,
        )
        self._buffer.setup_entry("observations", np.float32)
        self._buffer.setup_entry("actions", np.int32)
        self._buffer.setup_entry("logprobs", np.float32)
        self._buffer.setup_entry("rewards", np.float32)
        self._buffer.setup_entry("dones", bool)

        # Counters
        self._total_timesteps = 0
        self._total_updates = 0
        self._timesteps_since_last_update = 0

    @property
    def total_timesteps(self):
        return self._total_timesteps

    @property
    def parameters(self):
        return list(self._policy_net.parameters()) + list(self._value_net.parameters())

    def _get_value(self, observation):
        observation = torch.FloatTensor(observation).to(self._device)
        return self._value_net(observation)

    def _policy(self, observation, action=None):
        observation = torch.FloatTensor(observation).to(self._device)

        logits = self._policy_net(observation)
        action_dist = Categorical(logits=logits)

        if action is None:
            action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)
        return action, action_logprob, action_dist.entropy()

    def policy(self, observation):
        return self._policy(observation)[0].cpu().numpy()

    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.

        Parameters
        ----------
        budget: int
            Number of timesteps to train the agent for.
            One step = one transition in the environment.
        """
        del kwargs
        observation = self.env.reset()
        num_evals = 0
        self._start_time = time.time()

        while self._total_timesteps < budget:
            self._timesteps_since_last_update += self.num_envs
            with torch.no_grad():
                action, logprobs, _ = self._policy(observation)
            next_obs, reward, done, info = self.env.step(action.cpu().numpy())

            # Store data
            self._buffer.append(
                {
                    "observations": observation,
                    "actions": action,
                    "logprobs": logprobs,
                    "rewards": reward,
                    "dones": done,
                }
            )

            # Timestep counter and next observation
            self._total_timesteps += self.num_envs
            observation = next_obs

            # Log episode data
            if isinstance(info, dict):
                info = (info,)
            for env_info in info:
                if "episode" in env_info.keys():
                    print(
                        "Total Timesteps={}, Episode Return={}".format(
                            self._total_timesteps, env_info["episode"]["r"]
                        )
                    )
                    self.writer.add_scalar(
                        "episode_return",
                        env_info["episode"]["r"],
                        self._total_timesteps,
                    )
                    self.writer.add_scalar(
                        "episode_length",
                        env_info["episode"]["l"],
                        self._total_timesteps,
                    )
                    break  # solution from CleanRL, not ideal

            # Update (if the buffer is full)
            if self._buffer.full():
                self._update()

            # Evaluation
            total_timesteps = self._total_timesteps
            while (
                self._eval_interval is not None
                and total_timesteps // self._eval_interval >= num_evals
            ):
                eval_rets = self.eval(
                    n_simulations=self._eval_episodes,
                    eval_horizon=self._max_episode_steps,
                    gamma=1.0,
                )
                num_evals += 1
                if self.writer:
                    self.writer.add_scalar(
                        "evaluation_return", eval_rets, total_timesteps
                    )

    def _update(self):
        """Update networks."""
        # Get batch
        batch = self._buffer.get()

        clipfracs = []
        batch_size = self.num_steps * self.num_envs
        indices = np.arange(batch_size)  # used to create minibatches

        for _ in range(self.num_epochs):
            # Compute targets (at every epoch to avoid stale targets)
            with torch.no_grad():
                values = self._get_value(batch.data["observations"])
            discounts = self.gamma * (1.0 - batch.data["dones"])
            next_values = torch.roll(values, -1, 0)
            next_values[-1] = 0.0

            # TODO: implement naive returns for self.use_gae = False
            returns = lambda_returns(
                batch.data["rewards"], discounts, next_values, self.gae_lambda
            )
            batch.data["returns"] = returns
            batch.data["advantages"] = returns - values

            # Unroll batch
            unrolled_batch = {}
            for tag in batch.data.keys():
                shape = batch.data[tag].shape
                new_shape = (shape[0] * shape[1],) + shape[2:]
                unrolled_batch[tag] = batch.data[tag].reshape(new_shape)

            # Shuffle minibatches
            np.random.shuffle(indices)

            # Minibatch loop
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = indices[start:end]

                mb_obs = unrolled_batch["observations"][minibatch_indices]
                mb_acts = unrolled_batch["actions"][minibatch_indices]
                mb_logprobs = unrolled_batch["logprobs"][minibatch_indices]
                mb_returns = unrolled_batch["returns"][minibatch_indices]
                mb_advantages = unrolled_batch["advantages"][minibatch_indices]

                _, newlogprob, entropy = self._policy(mb_obs, mb_acts.long())
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                # Policy loss
                if self.normalize_advantage:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                # TODO: add AVEC and PPO-style clipping
                newvalues = self._get_value(mb_obs)
                vf_loss = 0.5 * ((newvalues - mb_returns) ** 2).mean()

                # Regularization loss
                reg_loss = entropy.mean()

                # Take gradient step
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * reg_loss

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
                self._optimizer.step()

            # target_kl early stopping
            if self.target_kl is not None:
                if approx_kl.item() > self.target_kl:
                    break

        # Update counters
        self._timesteps_since_last_update = 0
        self._total_updates += 1

        # Logging
        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        step = self._total_timesteps
        self.writer.add_scalar(
            "learning_rate", self._optimizer.param_groups[0]["lr"], step
        )
        self.writer.add_scalar("value_loss", vf_loss.item(), step)
        self.writer.add_scalar("policy_loss", pg_loss.item(), step)
        self.writer.add_scalar("entropy", reg_loss.item(), step)
        self.writer.add_scalar("old_approx_kl", old_approx_kl.item(), step)
        self.writer.add_scalar("approx_kl", approx_kl.item(), step)
        self.writer.add_scalar("clipfrac", np.mean(clipfracs), step)
        self.writer.add_scalar("explained_variance", explained_var, step)
        self.writer.add_scalar(
            "SPS", int(step / (time.time() - self._start_time)), step
        )

        print("SPS:", int(step / (time.time() - self._start_time)))

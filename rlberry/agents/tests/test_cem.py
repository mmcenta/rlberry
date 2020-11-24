from rlberry.agents.cem import CEMAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env


def test_cem_agent():
    env = get_benchmark_env(level=1)
    n_episodes = 5
    batch_size = 100
    horizon = 30
    gamma = 0.99

    agent = CEMAgent(env,
                     n_episodes,
                     horizon,
                     gamma,
                     batch_size,
                     percentile=70,
                     learning_rate=0.01)
    agent.fit()

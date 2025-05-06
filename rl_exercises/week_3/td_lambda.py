from __future__ import annotations

from typing import Any, DefaultDict

from collections import defaultdict

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_3 import EpsilonGreedyPolicy


class TDLambdaAgent(AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        policy: EpsilonGreedyPolicy,
        alpha: float = 0.5,
        gamma: float = 0.99,
        lambda_: float = 0.9,
    ) -> None:
        assert 0 <= gamma <= 1, "Gamma must be in [0, 1]"
        assert 0 <= lambda_ <= 1, "Lambda must be in [0, 1]"
        assert alpha > 0, "Alpha must be > 0"

        self.env = env
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_

        # number of actions → used by Q’s default factory
        self.n_actions = env.action_space.n

        # Build Q so that unseen states map to zero‐vectors
        self.Q: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )
        self.E: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

    def predict_action(self, state: np.array, evaluate: bool = False) -> Any:
        return self.policy(self.Q, state, evaluate=evaluate)

    def save(self, path: str) -> Any:
        np.save(path, self.Q)

    def load(self, path) -> Any:
        self.Q = np.load(path)

    def reset_traces(self) -> None:
        self.E = defaultdict(lambda: np.zeros(self.n_actions))

    def update_agent(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        next_action: int,
        done: bool,
    ) -> float:
        td_target = reward
        if not done:
            td_target += self.gamma * self.Q[next_state][next_action]
        td_error = td_target - self.Q[state][action]

        # Update eligibility trace
        self.E[state][action] += 1  # accumulating traces

        # Update all Q-values using eligibility traces
        for s in self.E:
            self.Q[s] += self.alpha * td_error * self.E[s]
            self.E[s] *= self.gamma * self.lambda_

        return self.Q[state][action]

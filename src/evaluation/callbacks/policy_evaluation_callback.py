# source: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

import config as config
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from simulation import simulate_policy
from market import Market

class PolicyEvaluationCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, evaluator, n_steps, verbose=0):

        self.evaluator = evaluator
        self.n_steps = n_steps
        self.last_time_trigger = 0

        self.customers = Market().customers

        self.prices = {s: { 'mean' : [], 'std' : [] } for s in range(config.week_length) }
        self.rewards = []

        super(PolicyEvaluationCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:

        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps

            try:
                state = np.array(self.locals['obs_tensor'][0])
            except KeyError:
                state = np.array(self.locals['new_obs'][0])

            infos = simulate_policy(self.model)

            reward = np.sum(infos[f'i0_total_reward'][int(config.episode_length/2):] + infos[f'i1_total_reward'][int(config.episode_length/2):])
            self.rewards.append(reward)

            [self.prices[s]['mean'].append(self.model.predict(np.array([s, *state[1:]]), deterministic=True)[0][0]) for s in range(config.week_length)]
            
            ## DQN ##
            # [self.prices[s]['mean'].append(self.model.predict(np.array([s, *state[1:]]), deterministic=True)[0]) for s in range(config.week_length)]

            prices_sample = []
            for _ in range(100):
                prices_sample.append([self.model.predict(np.array([s, *state[1:]]), deterministic=False)[0][0] for s in range(config.week_length)])
                
                ## DQN ##
                # prices_sample.append([self.model.predict(np.array([s, *state[1:]]), deterministic=False)[0] for s in range(config.week_length)])

            [self.prices[s]['std'].append(np.std(prices_sample, axis=0)[s]) for s in range(config.week_length)]


        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """

        self.evaluator.plot_prices(self.prices)
        self.evaluator.plot_rewards(self.rewards)

        infos = simulate_policy(self.model)
        self.evaluator.print_simulation_statistics(infos)
        
        pass

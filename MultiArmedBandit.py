import numpy as np


class Bandit(object):
    def __init__(self, **kwargs):
        """
        Create a single bandit with a function to generate a random value

        :param kwargs: Parameters to pass to the distribution function.
        """
        self._params = kwargs

    def get_value(self):
        return np.random.exponential(scale=self._params["scale"], size=1)[0]
        # return np.random.normal(loc=self._params["loc"], scale=self._params["scale"], size=1)[0]

    def get_param(self, key):
        return self._params[key]


class MultiArmedBandit(object):
    def __init__(self, n):
        """
        Create a multi-armed bandit with n bandits.

        Each bandit is stored as a set of parameters

        Each bandit has an exponentially distributed reward function.
        Parameters are selected from an exponential distribution
        :param n:
        """
        # Generate random bandits
        self._bandits = []
        for ii in range(n):
            # mu = np.random.exponential(scale=1, size=1)[0]
            # sigma = np.random.exponential(scale=0.5, size=1)[0]
            # self._bandits.append(Bandit(loc=mu, scale=sigma))
            scale = np.random.exponential(scale=1, size=1)[0]
            self._bandits.append(Bandit(scale=scale))

    def get_bandits(self):
        return self._bandits

    def choose(self, n):
        """
        Select bandit n and retrieve a reward
        :param n:
        :return:
        """
        return self._bandits[n].get_value()

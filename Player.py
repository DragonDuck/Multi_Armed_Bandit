import random


class Player(object):
    def __init__(self, bandits, action_selector="greedy", exploration_epsilon=0.1):
        """
        Create a player bot who implements an action strategy
        """
        self._bandits = bandits

        # Initialize empty action values for each bandit
        self._action_values = [0] * len(bandits.get_bandits())

        # Record how often each action was performed
        self._action_k = [0] * len(bandits.get_bandits())

        # Set the action selector
        if action_selector == "greedy":
            self._select_action = self.select_greedy_optimal_action
        elif action_selector == "exploratory":
            self._select_action = self.select_greedy_optimal_action_with_randomness
        else:
            self._select_action = self.select_greedy_optimal_action
            print("WARNING: Action selection algorithm '{}' not found. Using greedy selection".format(action_selector))

        # Various other parameters
        self._exploration_epsilon = exploration_epsilon

    def do_action(self):
        """
        Performs an action and updates internal parameters accordingly
        :return:
        """
        action = self._select_action()

        # Get reward
        reward = self._bandits.choose(n=action)

        # Update action values and counts
        self._action_values[action] = self.update_action_value(action=action, reward=reward)
        self._action_k[action] += 1

    def select_greedy_optimal_action(self):
        """
        Select the action with the highest current action value
        :return:
        """
        # select one of the maximum values at random
        max_indices = [
            i for i in range(len(self._action_values))
            if self._action_values[i] == max(self._action_values)]
        max_index = random.choice(max_indices)
        return max_index

    def select_greedy_optimal_action_with_randomness(self):
        """
        Select the action with the highest current action value but allow for occasional exploration
        :return:
        """
        if random.random() < self._exploration_epsilon:
            # If exploring, choose any of the indices that are NOT the greedy options
            possible_indices = [
                i for i in range(len(self._action_values))
                if self._action_values[i] != max(self._action_values)]
            # If all action values are identical, select them all anyway
            if len(possible_indices) == 0:
                possible_indices = [i for i in range(len(self._action_values))]
        else:
            possible_indices = [
                i for i in range(len(self._action_values))
                if self._action_values[i] == max(self._action_values)]

        # select one of the possible values at random
        return random.choice(possible_indices)

    def update_action_value(self, action, reward):
        """
        Incrementally updates the action value.

        This is an API function that allows variable algorithms to be plugged in.
        :param action: Index of the bandit
        :param reward: Actual reward obtained
        :return: The new value of the action
        """
        return self.update_action_value_incrementally(action=action, reward=reward)

    def update_action_value_incrementally(self, action, reward):
        """
        Incrementally update the action value based. The increment step size is 1/k where k is the number of times an
        action has been chosen.
        :param action: Index of the bandit
        :param reward: Actual reward obtained
        :return: The new value of the action
        """
        alpha = 1 / self._action_k[action] if self._action_k[action] > 0 else 1
        old_val = self._action_values[action]
        new_val = old_val + alpha * (reward - old_val)
        return new_val

    def get_action_values(self):
        return self._action_values

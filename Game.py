from MultiArmedBandit import MultiArmedBandit
from Player import Player


def run_stationary_game(num_games=1000, num_bandits=10):
    bandits = MultiArmedBandit(n=num_bandits)
    player = Player(bandits=bandits, action_selector="exploratory", exploration_epsilon=0.1)

    # Play the game N times
    for ii in range(num_games):
        player.do_action()

    print("Action Value Comparison")
    for ii in range(len(bandits.get_bandits())):
        bandit = bandits.get_bandits()[ii]
        val = player.get_action_values()[ii]
        print("Bandit {}: player_val = {:.3f}, mu = {:.3f}, sigma = {:.3f}".format(
            ii, val, bandit.get_param("loc"), bandit.get_param("scale")))

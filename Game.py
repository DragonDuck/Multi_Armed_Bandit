from MultiArmedBandit import MultiArmedBandit
from Player import Player
import matplotlib.pyplot as plt


def run_stationary_game(bandits, num_games=1000):
    player_greedy = Player(bandits=bandits, action_selector="greedy")
    player_explo = Player(bandits=bandits, action_selector="exploratory", exploration_epsilon=0.1)

    # Play the game N times for each player and keep track of what they earn
    for ii in range(num_games):
        player_greedy.do_action()
        player_explo.do_action()

    return player_greedy.get_total_earnings(), player_explo.get_total_earnings()


bandits = MultiArmedBandit(n=10)

# Run through different game lengths
scores_greedy = []
scores_explo = []
num_games = [10, 25, 50, 100, 250, 500, 1000]
for ng in num_games:
    # Average over multiple iterations
    greedy = explo = 0
    num_iterations = 100
    for ii in range(num_iterations):
        new_greedy, new_explo = run_stationary_game(bandits=bandits, num_games=ng)
        greedy += new_greedy
        explo += new_explo
    scores_greedy.append(greedy / (num_iterations * ng))
    scores_explo.append(explo / (num_iterations * ng))

plt.plot(num_games, scores_greedy, num_games, scores_explo)

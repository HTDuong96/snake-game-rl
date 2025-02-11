"""
Author: Son Phat Tran
"""
from utils.graph import plot_training_progress
from config import CELL_SIZE

from game import SnakeGame

from reinforcement_learning import play_game as play_game_ai, train_ai
from graph import play_game as play_game_graph


def train_and_play_rl():
    # Train the agent
    trained_agent, training_scores = train_ai(render_game=False, episodes=1001)

    # Plot the scores
    plot_training_progress(training_scores)

    # Play a few games with the trained agent
    play_game_ai(trained_agent)


def play_graph():
    play_game_graph(SnakeGame(cell_size=CELL_SIZE, is_rl=False))


if __name__ == "__main__":
    train_and_play_rl()

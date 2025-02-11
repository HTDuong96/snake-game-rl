"""
Author: Son Phat Tran
"""
from graph.hamiltonian_cycle import HamiltonianAgent


def play_game(game):
    agent = HamiltonianAgent(game.game_width, game.game_height)

    # Game loop
    while True:
        state = game._get_state()
        action = agent.get_action(state, game)

        _, reward, done = game.step(action)
        game.render()

        if done:
            break

    game.close()

"""
Author: Son Phat Tran
"""
import pygame
import torch

from game.snake_game import SnakeGame
from reinforcement_learning.snake_ai_rl import SnakeReinforcementLearningAI

from config import BOARD_SIZE, CELL_SIZE


def train_ai(render_game=True, episodes=1000):
    env = SnakeGame(game_width=BOARD_SIZE, game_height=BOARD_SIZE, cell_size=CELL_SIZE)
    agent = SnakeReinforcementLearningAI()
    scores = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return agent, scores

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()

            if render_game:
                env.render()

            state = next_state
            total_reward += reward

        scores.append(env.score)

        if episode % 100 == 0:
            print(f"Episode: {episode}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")
            print("Saving model ...")
            agent.save(episode)

    return agent, scores


def play_game(agent, num_games=5):
    """Play the game using a trained agent"""
    env = SnakeGame(game_width=BOARD_SIZE, game_height=BOARD_SIZE, cell_size=CELL_SIZE)

    for game in range(num_games):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            # Get action from trained agent
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = agent.q_network(state).argmax().item()

            # Take step
            state, reward, done = env.step(action)
            total_reward += reward

            # Render game
            env.render()

        print(f"Game {game + 1} Score: {env.score}")

    env.close()


def load_and_play(model_path="snake_ai_model.pth", num_games=5):
    """
    Load a trained model and play the Snake game

    Args:
        model_path (str): Path to the saved model weights
        num_games (int): Number of games to play
    """
    # Initialize agent and load weights
    agent = SnakeReinforcementLearningAI()
    try:
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.q_network.eval()  # Set to evaluation mode
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize game environment
    env = SnakeGame(game_width=BOARD_SIZE, game_height=BOARD_SIZE, cell_size=CELL_SIZE)

    # Track scores
    scores = []

    for game in range(num_games):
        state = env.reset()
        done = False

        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return scores

            # Get action from trained agent
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = agent.q_network(state_tensor).argmax().item()

            # Take step
            state, reward, done = env.step(action)

            # Render game
            env.render()

            if done:
                scores.append(env.score)
                print(f"Game {game + 1} Score: {env.score}")

    env.close()
    return scores

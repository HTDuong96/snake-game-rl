"""
Author: Son Phat Tran (Modified for centered display)
"""
import numpy as np
import pygame
import random

from config import GAME_SPEED


class SnakeGame:
    def __init__(self, game_width=20, game_height=20, cell_size=20, is_rl=True):
        # Check if reinforcement learning
        self.is_rl = is_rl

        # Game grid dimensions
        self.game_width = game_width
        self.game_height = game_height
        self.cell_size = cell_size

        # Calculate game surface dimensions
        self.game_surface_width = game_width * cell_size
        self.game_surface_height = game_height * cell_size

        # Initial window size (larger than game surface)
        self.window_width = 1024  # Initial width
        self.window_height = 768  # Initial height

        # Initialize Pygame
        pygame.init()
        self.window = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        self.game_surface = pygame.Surface((self.game_surface_width, self.game_surface_height))
        pygame.display.set_caption('Snake AI')

        # Initialize font with antialiasing
        self.font = pygame.font.SysFont("helveticaneue", 48)
        self.font.set_bold(True)  # Make the font bold for better visibility

        # Initial offset calculation
        self._update_offsets()
        self.clock = pygame.time.Clock()

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.HEAD_COLOR = (0, 200, 255)  # Light blue for snake's head
        self.BACKGROUND_COLOR = (50, 50, 50)  # Dark gray for the window background

        self.reset()

    def reset(self):
        self.snake = [(self.game_width // 2, self.game_height // 2)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, self.game_width - 1), random.randint(0, self.game_height - 1))
            if food not in self.snake:
                return food

    def _get_state(self):
        head = self.snake[0]

        # Get the points around the head
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)

        # Current direction
        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        def count_empty_neighbors(pos):
            neighbors = [
                (pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]),
                (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)
            ]
            return sum(1 for n in neighbors if not self._is_collision(n))

        # Check spaces two steps ahead
        next_pos = (head[0] + self.direction[0], head[1] + self.direction[1])
        future_space = count_empty_neighbors(next_pos) if not self._is_collision(next_pos) else 0

        state = [
            # Danger detection (8 directions)
            self._is_collision((head[0] - 1, head[1] - 1)),  # Danger top-left
            self._is_collision((head[0], head[1] - 1)),  # Danger top
            self._is_collision((head[0] + 1, head[1] - 1)),  # Danger top-right
            self._is_collision((head[0] - 1, head[1])),  # Danger left
            self._is_collision((head[0] + 1, head[1])),  # Danger right
            self._is_collision((head[0] - 1, head[1] + 1)),  # Danger bottom-left
            self._is_collision((head[0], head[1] + 1)),  # Danger bottom
            self._is_collision((head[0] + 1, head[1] + 1)),  # Danger bottom-right

            # Current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food direction
            self.food[0] < head[0],  # Food left
            self.food[0] > head[0],  # Food right
            self.food[1] < head[1],  # Food up
            self.food[1] > head[1],  # Food down

            # Food distance (normalized)
            abs(self.food[0] - head[0]) / self.game_width,
            abs(self.food[1] - head[1]) / self.game_height,

            # Distance to walls (normalized)
            head[0] / self.game_width,  # Distance to right wall
            (self.game_width - head[0]) / self.game_width,  # Distance to left wall
            head[1] / self.game_height,  # Distance to bottom wall
            (self.game_height - head[1]) / self.game_height,  # Distance to top wall

            # Snake body nearby (in adjacent cells)
            any(segment == point_l for segment in self.snake[1:]),  # Body left
            any(segment == point_r for segment in self.snake[1:]),  # Body right
            any(segment == point_u for segment in self.snake[1:]),  # Body up
            any(segment == point_d for segment in self.snake[1:]),  # Body down

            # Future space information (normalized)
            future_space / 4.0,  # Number of available moves in next position

            # Body awareness
            sum(1 for segment in self.snake[1:]
                if abs(head[0] - segment[0]) + abs(head[1] - segment[1]) <= 2) / 8.0,
        ]
        return np.array(state, dtype=float)

    def _is_collision(self, pos):
        return (pos[0] < 0 or pos[0] >= self.game_width or
                pos[1] < 0 or pos[1] >= self.game_height or
                pos in self.snake[:-1])

    def _turn_right(self, direction):
        return (-direction[1], direction[0])

    def _turn_left(self, direction):
        return (direction[1], -direction[0])

    def step(self, action):
        # Actions: [straight, right, left]
        if action == 1:
            self.direction = self._turn_right(self.direction)
        elif action == 2:
            self.direction = self._turn_left(self.direction)

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Check if game is over (collision)
        done = self._is_collision(new_head)
        if done:
            if new_head in self.snake:
                return self._get_state(), -100, True
            return self._get_state(), -10, True

        # Move snake
        self.snake.insert(0, new_head)

        # Calculate base reward based on distance to food
        old_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        distance_reward = old_distance - new_distance

        # Check if food is eaten
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 20 + min(len(self.snake), 10)
            self.food = self._place_food()
        else:
            self.snake.pop()
            reward = distance_reward

            if self._is_near_body(new_head):
                reward -= 1

        self.steps += 1

        if self.is_rl and self.steps > 100 * len(self.snake):
            done = True
            reward = -5

        return self._get_state(), reward, done

    def _is_near_body(self, pos):
        for segment in self.snake[1:]:
            if abs(pos[0] - segment[0]) + abs(pos[1] - segment[1]) <= 1:
                return True
        return False

    def _update_offsets(self):
        """Update the offsets when window is resized to keep game centered"""
        self.window_width, self.window_height = self.window.get_size()

        # Reserve space for score text above game (70 pixels for text area)
        score_height = 70
        available_height = self.window_height - score_height

        # Calculate offsets to center game in remaining space
        self.offset_x = (self.window_width - self.game_surface_width) // 2
        self.offset_y = score_height + (available_height - self.game_surface_height) // 2

    def render(self):
        # Handle window resize events
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                self.window = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self._update_offsets()
            elif event.type == pygame.QUIT:
                self.close()
                return

        # Fill window background
        self.window.fill(self.BACKGROUND_COLOR)

        # Fill game surface background
        self.game_surface.fill(self.BLACK)

        # Draw snake body on game surface
        for segment in self.snake[1:]:
            pygame.draw.rect(self.game_surface, self.GREEN,
                           (segment[0] * self.cell_size,
                            segment[1] * self.cell_size,
                            self.cell_size - 2,
                            self.cell_size - 2))

        # Draw head on game surface
        head = self.snake[0]
        pygame.draw.rect(self.game_surface, self.HEAD_COLOR,
                        (head[0] * self.cell_size,
                         head[1] * self.cell_size,
                         self.cell_size - 2,
                         self.cell_size - 2))

        # Draw food on game surface
        pygame.draw.rect(self.game_surface, self.RED,
                        (self.food[0] * self.cell_size,
                         self.food[1] * self.cell_size,
                         self.cell_size - 2,
                         self.cell_size - 2))

        # Draw game surface on window at offset position
        self.window.blit(self.game_surface, (self.offset_x, self.offset_y))

        # Draw score text centered above the game area
        score_text = self.font.render(f'{self.score}', True, self.WHITE)
        text_rect = score_text.get_rect()
        text_rect.centerx = self.window_width // 2
        text_rect.top = 20  # 20 pixels from top of window
        self.window.blit(score_text, text_rect)

        # Draw border around game area
        pygame.draw.rect(self.window, self.WHITE,
                        (self.offset_x - 2, self.offset_y - 2,
                         self.game_surface_width + 4, self.game_surface_height + 4),
                        2)

        pygame.display.flip()
        self.clock.tick(GAME_SPEED)

    def close(self):
        pygame.quit()
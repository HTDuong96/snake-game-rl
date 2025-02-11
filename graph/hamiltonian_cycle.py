def generate_hamiltonian_cycle(n=20):
    """
    Generates a Hamiltonian cycle on an n x n grid using a snake-like pattern.
    Returns a list of (x, y) coordinates representing the path.
    """
    path = []

    # Generate the snake pattern going down
    for x in range(n):
        if x % 2 == 0:
            # Going up
            for y in range(n):
                path.append((x, y))
        else:
            # Going down
            for y in range(n - 1, -1, -1):
                path.append((x, y))

    # Connect back to the start:
    # We're now at (n-1, 0) or (n-1, n-1) depending on whether n is even or odd
    # We need to traverse back to (0, 0) along the bottom edge
    # If we ended in the bottom-right corner
    if (n - 1) % 2 == 0:
        # Move left along bottom edge
        for x in range(n - 2, -1, -1):
            path.append((x, 0))
    else:
        # Move to bottom edge first
        path.append((n - 1, 0))
        # Move left along bottom edge
        for x in range(n - 2, -1, -1):
            path.append((x, 0))

    return path


class HamiltonianAgent:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Generate the Hamiltonian circuit path
        self.path = generate_hamiltonian_cycle(n=width)

        # Create a lookup dictionary for positions in the path
        self.path_indices = {pos: idx for idx, pos in enumerate(self.path)}

    def get_action(self, state, snake_game):
        """Determine the next action based on the Hamiltonian circuit."""
        head = snake_game.snake[0]
        current_idx = self.path_indices[head]

        # Get the next position in the circuit
        next_idx = (current_idx + 1) % len(self.path)
        target = self.path[next_idx]

        # Calculate the direction we need to move
        dx = target[0] - head[0]
        dy = target[1] - head[1]
        target_direction = (dx, dy)

        # Current direction of the snake
        current_direction = snake_game.direction

        # If we're already facing the right direction, go straight
        if target_direction == current_direction:
            return 0  # Straight

        # Determine if we need to turn left or right
        # Turn: (-y, x)
        # Turn left: (y, -x)
        right_turn = (-current_direction[1], current_direction[0])
        if target_direction == right_turn:
            return 1  # Right
        else:
            return 2  # Left

    def shortcut_to_food(self, snake_game):
        """
        Check if we can safely take a shortcut to the food.
        Currently, always returns False for guaranteed safety.
        """
        return False  # Default to safe behavior
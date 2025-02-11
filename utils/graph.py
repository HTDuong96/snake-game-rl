"""
Author: Son Phat Tran
"""

import numpy as np
from matplotlib import pyplot as plt


def plot_training_progress(scores, window_size=100):
    """
    Plot the training progress showing both raw scores and running average

    Args:
        scores (list): List of training scores
        window_size (int): Size of the moving average window
    """
    # Convert to numpy array for easier manipulation
    scores_array = np.array(scores)

    # Calculate running average
    running_avg = np.zeros(len(scores_array))
    for i in range(len(scores_array)):
        if i < window_size:
            running_avg[i] = np.mean(scores_array[:i + 1])
        else:
            running_avg[i] = np.mean(scores_array[i - window_size + 1:i + 1])

    # Create figure and axis objects with a single subplot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw scores with low opacity
    ax.plot(scores_array, alpha=0.3, color='blue', label='Raw Scores')

    # Plot running average
    ax.plot(running_avg, color='red', linewidth=2, label=f'{window_size}-Episode Running Average')

    # Add labels and title
    ax.set_title('Training Progress', fontsize=15, pad=15)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax.legend()

    # Add score statistics
    stats_text = f'Max Score: {np.max(scores_array):.1f}\n'
    stats_text += f'Average Score: {np.mean(scores_array):.1f}\n'
    stats_text += f'Final {window_size}-Episode Average: {running_avg[-1]:.1f}'

    plt.text(0.02, 0.98, stats_text,
             transform=ax.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Adjust layout to prevent text cutoff
    plt.tight_layout()

    # Show plot
    plt.show()

    # Save graph
    fig.savefig("chart/training_score.png")

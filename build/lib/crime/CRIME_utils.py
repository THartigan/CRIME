
import tensorflow as tf
import numpy as np

# Helper functions within functions

def mean_relative_percentage_error(y_true, y_pred):
    # Avoid division by zero
    denominator = tf.where(tf.math.equal(y_true, 0), tf.ones_like(y_true), y_true)
    
    # Calculate relative percentage error
    rpe = tf.abs((y_pred - y_true) / denominator)
    
    # Return mean relative percentage error
    return 100 * tf.reduce_mean(rpe)


def cosine_similarity_manual(v1, v2):

    """
    Calculate cosine similarity between spectra.

    Parameters:
    - v1: array of spectra 1
    - v1: array of spectra 2

    Returns:
    - scos: cosine similarity
    """

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    scos = dot_product / (norm_v1 * norm_v2)
    return scos


def discretize_concentration(concentrations, thresholds=(3, 6)):
    """
    Discretize continuous concentration levels into categorical bins.

    Parameters:
    - concentrations: array of concentration values
    - thresholds: tuple containing the thresholds for categorization

    Returns:
    - discretized_labels: array of discretized concentration levels
    """

    discretized_labels = []
    for concentration in concentrations:
        if concentration == 0:
            label = 0  # No concentration
        elif concentration <= thresholds[0]:
            label = 1  # Low concentration
        elif concentration <= thresholds[1]:
            label = 2  # Medium concentration
        else:
            label = 3  # High concentration
        discretized_labels.append(label)
    
    return discretized_labels

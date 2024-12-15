import numpy as np
import time

# from scipy.ndimage import convolve

# Generate a synthetic input matrix (example: 6x6 image)
input_matrix = np.random.rand(1000, 1000)

# Define a synthetic filter (example: 3x3 filter)
filter_matrix = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])


def convolve_2d(input_matrix, filter_matrix, stride):
    """Perform 2D convolution with stride."""
    input_size = input_matrix.shape[0]
    filter_size = filter_matrix.shape[0]
    output_size = (input_size - filter_size) // stride + 1
    output = np.zeros((output_size, output_size))

    for i in range(0, output_size):
        for j in range(0, output_size):
            region = input_matrix[
                i * stride : i * stride + filter_size,
                j * stride : j * stride + filter_size,
            ]
            output[i, j] = np.sum(region * filter_matrix)

    return output


def max_pooling(matrix, pool_size, stride):
    """Perform max pooling."""
    input_size = matrix.shape[0]
    output_size = (input_size - pool_size) // stride + 1
    pooled = np.zeros((output_size, output_size))

    for i in range(0, output_size):
        for j in range(0, output_size):
            region = matrix[
                i * stride : i * stride + pool_size, j * stride : j * stride + pool_size
            ]
            pooled[i, j] = np.max(region)

    return pooled


# Evaluate different stride values
strides = [1, 2, 3, 4]
results = []

for stride in strides:
    # Perform convolution
    start = time.time()
    conv_output = convolve_2d(input_matrix, filter_matrix, stride)
    end = time.time()
    # Perform max pooling (example: 2x2 pool with stride=2)
    pooled_output = max_pooling(conv_output, pool_size=2, stride=2)
    # Calculate variance as a proxy for feature richness
    variance = np.var(pooled_output)
    results.append((stride, variance, (end - start) * 10000))

# Find the stride with the highest variance
optimal_stride = max(results, key=lambda x: x[1])

print("Stride results (stride, variance, time):", results)
print("Optimal stride:", optimal_stride[0])

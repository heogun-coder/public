import numpy as np
import time


filter_matrix = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # KERNAL


def convolve_2d(input_matrix, filter_matrix, stride):  # 합성곱
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


def max_pooling(matrix, pool_size, stride):  # 최대 풀링
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


res = [0, 0, 0, 0]
ti = [0, 0, 0, 0]
for x in range(100):

    input_matrix = np.random.rand(300, 300)  # INPUT

    strides = [1, 2, 3, 4]
    results = []

    for stride in strides:

        start = time.time()
        conv_output = convolve_2d(input_matrix, filter_matrix, stride)
        end = time.time()
        pooled_output = max_pooling(conv_output, pool_size=2, stride=2)
        t = end - start
        variance = np.var(pooled_output)
        # results.append((stride, variance.round(5), round(t, 5)))

        res[stride - 1] += variance.round(5)
        ti[stride - 1] += round(t, 5)

for x in range(0, 4):
    print(f"stride = {x + 1}")
    print(f"variance : {res[x]}, time : {ti[x]}")


# print("Stride results (stride, variance, time):", results)
# print("Optimal stride:", optimal_stride[0])

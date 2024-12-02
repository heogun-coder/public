def generate_key(matrix_size, matrix):

    L = matrixify(matrix_size)
    for x in range(matrix_size):
        L[x][x] = 1

    U = matrixify(matrix_size)
    for x in range(matrix_size):
        for y in range(matrix_size):
            U[x][y] = matrix[x][y]

    for x in range(matrix_size):
        for y in range(0, x):
            if U[x][y] != 0:
                div = U[x][y] / U[y][y]
                L[x][y] = div
                for z in range(y, matrix_size):
                    U[x][z] -= U[y][z] * div
    det = 1
    for x in range(matrix_size):
        det *= U[x][x]

    return det, L, U


def matrixify(matrix_size):
    matrix = [0 for _ in range(matrix_size)]
    for i in range(matrix_size):
        matrix[i] = [0 for _ in range(matrix_size)]
    return matrix


def set_matrix(sentence):
    matrix_size = len(sentence)

    matrix = matrixify(matrix_size)

    return matrix


sentence = "helloworld"
key_public_det, key_public_matrix, key_private = generate_key(10, set_matrix(sentence))

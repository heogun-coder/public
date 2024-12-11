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
                div = int(U[x][y] / U[y][y])
                L[x][y] = div
                for z in range(y, matrix_size):
                    U[x][z] -= U[y][z] * div
    det = 1
    for x in range(matrix_size):
        det *= U[x][x]

    return det, L, U


def solve(key_public, key_private):
    m = len(key_public)
    n = len(key_public[0])  # == len(key_private)
    r = len(key_private[0])
    A = [0 for _ in range(m)]
    for x in range(m):
        A[x] = [0 for _ in range(r)]

    for x in range(m):
        for y in range(r):
            for z in range(n):
                A[x][y] += key_public[x][z] * key_private[z][y]

    return A


def matrixify(matrix_size):
    matrix = [0 for _ in range(matrix_size)]
    for i in range(matrix_size):
        matrix[i] = [0 for _ in range(matrix_size)]
    return matrix


def get_sqrt(n):
    i = 1
    while 1:
        if n / (i**2) <= 1:
            return i
        i += 1


def set_matrix(sentence):
    matrix_size = get_sqrt(len(sentence))
    matrix = matrixify(matrix_size)

    for x in range(matrix_size):
        for y in range(matrix_size):
            matrix[x][y] = sentence[matrix_size * x + y]  # ord함수 추가
    return matrix


def get_square_matrix(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def get_sub_matrix(matrix, i, j):
    sub_matrix = matrixify(len(matrix) - 1)
    for x in range(i):
        for y in range(j):
            sub_matrix[x][y] = matrix[x][y]
        for y in range(j + 1, len(matrix)):
            sub_matrix[x][y - 1] = matrix[x][y]

    for x in range(i, len(matrix)):
        for y in range(j):
            sub_matrix[x - 1][y] = matrix[x][y]
        for y in range(j + 1, len(matrix)):
            sub_matrix[x - 1][y - 1] = matrix[x][y]

    return sub_matrix


def det(matrix):
    matrix_size = len(matrix)
    result = 0
    y = 0
    index = 0
    if matrix_size > 2:
        for x in range(matrix_size):
            if (x + 1 + y + 1) % 2 == 0:
                index = 1
            else:
                index = -1
            result += (matrix[x][y]) * index * det(get_sub_matrix(matrix, x, y))
    elif matrix_size == 2:
        return get_square_matrix(matrix)

    return result


sentence = [1, 2, 3, 2, 3, 4, 6, 5, 7]  # 문자열 행렬로 변환
# 행렬곱 최적화
key_public_det, key_public_matrix, key_private = generate_key(
    get_sqrt(len(sentence)), set_matrix(sentence)
)
# 공개키의 임의의 원소를 0으로 처리 -> L' , 수신자는 A' = L'U 행렬 재구성
# 메세지의 원소에 행렬식 공개키 더하기. 이러고 공개키 곱하기 -> D생성
# D공개하면 수신자가 개인키 곱하고, A'의 역행렬을 곱해 메세지 복원하기.


print(key_public_matrix, key_private, key_public_det)


print(solve(key_public_matrix, key_private))
print(det(key_public_matrix), det(key_private))

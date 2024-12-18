import numpy as np


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


def LU_decomposition(A):
    """
    LU 분해 함수 (행렬 A를 L, U로 분해)
    A = L * U 형태로 반환.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1  # L의 대각선은 1로 설정

        # U의 첫 번째 행을 계산
        for j in range(i, n):
            U[i, j] = A[i, j] - np.sum(L[i, :i] * U[:i, j])

        # L의 첫 번째 열을 계산
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.sum(L[j, :i] * U[:i, i])) / U[i, i]

    return L, U


def forward_substitution(L, b):
    """
    L * x = b 형태의 선형 시스템을 풀어 x를 구하는 함수 (L은 하삼각행렬)
    """
    n = len(b)
    x = np.zeros_like(b)

    for i in range(n):
        x[i] = b[i] - np.sum(L[i, :i] * x[:i])

    return x


def backward_substitution(U, y):
    """
    U * x = y 형태의 선형 시스템을 풀어 x를 구하는 함수 (U는 상삼각행렬)
    """
    n = len(y)
    x = np.zeros_like(y)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(U[i, i + 1 :] * x[i + 1 :])) / U[i, i]

    return x


def inverse_matrix(A):
    L, U = LU_decomposition(A)
    n = A.shape[0]
    inv_A = np.zeros_like(A)

    # 단위행렬을 역행렬로 변환하는 방법
    for i in range(n):
        # 단위행렬의 i번째 열 벡터
        e_i = np.zeros(n)
        e_i[i] = 1

        # L * y = e_i를 풀어 y 구하기
        y = forward_substitution(L, e_i)

        # U * x = y를 풀어 x 구하기
        inv_A[:, i] = backward_substitution(U, y)

    return inv_A


def set_matrix(sentence):
    matrix_size = get_sqrt(len(sentence))
    matrix = matrixify(matrix_size)

    for x in range(matrix_size):
        for y in range(matrix_size):
            if type(sentence[matrix_size * x + y]) == int:
                matrix[x][y] = sentence[matrix_size * x + y]  # ord함수 추가
            elif type(sentence[matrix_size * x + y]) == str:
                matrix[x][y] = ord(sentence[matrix_size * x + y])
    return matrix


def encrypt_message(sentence, public_key, randomizer):

    temp = matrixify(len(public_key))

    for x in range(len(public_key)):
        for y in range(len(public_key[0])):
            public_key[x][y] += randomizer
            temp[x][y] = ord(sentence[x * len(public_key) + y])

    print(f"orded messeage : {temp}")
    print(f"public key : {public_key}")

    cipher = solve(temp, public_key)
    return cipher


def decrypt_message(cipher, private_key, matrix, public_det):
    result = solve(cipher, private_key)
    inv = inverse_matrix(matrix)
    result = solve(result, inv)
    for x in range(len(cipher)):
        for y in range(len(cipher[0])):
            result[x][y] -= public_det

    sentence = matrixify(len(cipher))
    for x in range(len(cipher)):
        for y in range(len(cipher[0])):
            sentence[x][y] = chr(result[x][y])

    return sentence


key = [[1, 2, 3], [2, 3, 4], [6, 5, 7]]  # 문자열 행렬로 변환
# 행렬곱 최적화
key_public_det, key_public_matrix, key_private = generate_key(len(key), key)
# 공개키의 임의의 원소를 0으로 처리 -> L' , 수신자는 A' = L'U 행렬 재구성
# 메세지의 원소에 행렬식 공개키 더하기. 이러고 공개키 곱하기 -> D생성
# D공개하면 수신자가 개인키 곱하고, A'의 역행렬을 곱해 메세지 복원하기.


print("keys : ", key_public_matrix, key_private, key_public_det)

# print(solve(key_public_matrix, key_private))  #!순서 중요

key_public_matrix[0][0] = 0

# print(solve(key_public_matrix, key_private))
key = solve(key_public_matrix, key_private)

message = "Ilikeyou!"

cipher = encrypt_message(message, key_public_matrix, key_public_det)
print(f"cipher : {cipher}")

sentence = decrypt_message(cipher, key_private, key, key_public_det)
print(f"sentence : {sentence}")

from math import sqrt
from key_certify import LU_decomposition, get_inverse


def generate_key():
    pass


def LU_decomposition1(A):
    L = [0 for _ in range(len(A))]
    U = [0 for _ in range(len(A))]
    for i in range(len(A)):
        L[i] = [0 for _ in range(len(A))]
        L[i][i] = 1
        U[i] = [0 for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(A)):
            U[i][j] = A[i][j]

    for i in range(len(A)):
        for j in range(0, i):
            if U[i][j] != 0:
                div = int(U[i][j] / U[j][j])
                L[i][j] = div
                for k in range(j, len(A)):
                    U[i][k] -= U[j][k] * div
    determinant = 1
    for x in range(len(A)):
        determinant *= U[x][x]
    return L, U, determinant


def check_key(key_public):
    if det(key_public) == 0:
        return True
    return False


def det(matrix):
    result = 0
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif len(matrix) > 2:
        for i in range(len(matrix)):
            result += matrix[0][i] * det(get_sub_matrix(matrix, 0, i)) * (-1) ** i
        return result

    return result


def get_sub_matrix(matrix, i, j):
    return [row[:j] + row[j + 1 :] for row in (matrix[:i] + matrix[i + 1 :])]


def summation(A, B, code):
    if code == 1:
        for x in range(len(A)):
            for y in range(len(A[x])):
                A[x][y] += B[x][y]
    elif code == -1:
        for x in range(len(A)):
            for y in range(len(A[x])):
                A[x][y] -= B[x][y]
    return A


def multiplication(A, B):
    result = [[0 for _ in range(len(A))] for _ in range(len(A))]
    for x in range(len(A)):
        for y in range(len(A[x])):
            for z in range(len(A[x])):
                result[x][y] += A[x][z] * B[z][y]
            # 소수점 둘째 자리 이하 버림
            if result[x][y] != int(result[x][y]):
                result[x][y] = float(format(result[x][y], ".2f"))
    return result


def print_matrix(matrix):
    for x in range(len(matrix)):
        print(matrix[x])


def encryption(key_public, message):
    result = multiplication(key_public, message)
    return result


def decryption(L, U, R, cipher):
    result = multiplication(cipher, U)  # cipher * U
    RU = multiplication(R, U)  # RU

    result = summation(result, RU, -1)  # cipher * U - RU

    I = [[0 for _ in range(len(result))] for _ in range(len(result))]
    for x in range(len(result)):
        I[x][x] = 1

    result = multiplication(result, get_inverse(RU))
    result = summation(result, I, 1)

    D1 = multiplication(L, U)  # A - R = LU
    D2 = get_inverse(RU)  # RU_inverse
    D = multiplication(D1, D2)
    D = summation(D, I, 1)

    result = multiplication(cipher, get_inverse(D))

    return result


"""
def get_inverse(matrix):
    L, U, _ = LU_decomposition(matrix)
    n = len(matrix)
    inverse = [[0 for _ in range(n)] for _ in range(n)]

    # 단위행렬 I를 만듭니다
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    # AX = I 를 풀기 위해 각 열마다 계산
    for k in range(n):
        # Step 1: Ly = b 풀기 (전진대입)
        y = [0 for _ in range(n)]
        b = [I[i][k] for i in range(n)]  # I의 k번째 열

        for i in range(n):
            sum = 0
            for j in range(i):
                sum += L[i][j] * y[j]
            y[i] = b[i] - sum  # L[i][i]는 1이므로 나눌 필요 없음

        # Step 2: Ux = y 풀기 (후진대입)
        x = [0 for _ in range(n)]
        for i in range(n - 1, -1, -1):
            sum = 0
            for j in range(i + 1, n):
                sum += U[i][j] * x[j]
            x[i] = (y[i] - sum) / U[i][i]

        # k번째 열에 결과 저장
        for i in range(n):
            inverse[i][k] = x[i]

    return inverse
"""


def to_matrix(message):
    result = [
        [0 for _ in range(int(sqrt(len(message))))]
        for _ in range(int(sqrt(len(message))))
    ]
    for x in range(len(message)):
        result[x // int(sqrt(len(message)))][x % int(sqrt(len(message)))] = ord(
            message[x]
        )
    return result


# key = generate_key(seed)
key = [[1, 2, 3], [2, 3, 4], [6, 5, 1]]

# print(det(key))

R = [[0, 2, 3], [2, 4, 6], [1, 1, 8]]
key_public_L, key_private, determinant = LU_decomposition1(key)
# L, U, det

# print_matrix(multiplication(key_public, key_private))

key_public = summation(key_public_L, R, 1)
# L+R
print("--key_public--")
print_matrix(key_public)

# A = LU + R
# key + R = A

message = "ILOVEYOU!"
message = to_matrix(message)

print("--message--")
print_matrix(message)


print("--encryption--")
cipher = encryption(key_public, message)
print_matrix(cipher)

print("--decryption--")
sentence = decryption(key_public_L, key_private, R, cipher)
print(sentence)

from math import sqrt


def generate_key():
    pass


def LU_decomposition(A):
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
    return result


def print_matrix(matrix):
    for x in range(len(matrix)):
        print(matrix[x])


def encryption(key_public, message):
    result = multiplication(key_public, message)
    return result


def decryption(A, U, R, cipher):
    result = multiplication(U, cipher)  # U * cipher
    RU = multiplication(R, U)  # RU
    result = summation(result, RU, -1)  # U*cipher - RU

    I = [[0 for _ in range(len(result))] for _ in range(len(result))]
    for x in range(len(result)):
        I[x][x] = 1
    result = summation(result, I, -1)

    D1 = summation(result, R, -1)  # A - R
    D2 = inverse(RU)  # RU_inverse
    D = multiplication(D1, D2)
    D = summation(D, I, 1)

    result = multiplication(cipher, inverse(D))
    return result


def inverse(matrix):
    result = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]
    pass

    return result


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
key_public, key_private, determinant = LU_decomposition(key)


# print_matrix(multiplication(key_public, key_private))

key_public = summation(key_public, R, 1)
print("--key_public--")
print_matrix(key_public)


message = "ILOVEYOU!"
message = to_matrix(message)

print("--message--")
print_matrix(message)


print("--encryption--")
cipher = encryption(key_public, message)
print_matrix(cipher)

# print("--decryption--")
# print_matrix(decryption(key_private, R, cipher))

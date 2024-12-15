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


def get_inverse(matrix):
    result = matrixify(len(matrix))

    return result


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


def decrypt_message(cipher, private_key):
    result = solve(cipher, private_key)

    pass

    return result


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


message = "Ilikeyou!"

cipher = encrypt_message(message, key_public_matrix, key_public_det)
print(f"cipher : {cipher}")

sentence = decrypt_message(cipher, key_private)
print(f"sentence : {sentence}")

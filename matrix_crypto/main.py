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
    return L, U


# key = generate_key(seed)
key = [1, 2, 3, 4, 5, 6, 7, 8, 9]
R = [0, 1, 4, 3, 2, 6, 5, 3, 2]

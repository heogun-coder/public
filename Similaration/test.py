def f1(x, y):
    return x**2 + y**2 - 4


def f1_diff(x, y):
    result = [2 * x, 2 * y]
    return result


def f2(x, y):
    return x * y - 4


def f2_diff(x, y):
    result = [y, x]
    return result


# -> 해 : x,y = 2,2 / x,y = -2,-2


def multiply_matrix(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for x in range(len(A)):
        for y in range(len(B[0])):
            for z in range(len(B)):
                result[x][y] += A[x][z] * B[z][y]
    return result


def LU_decomposition(A):
    L = [[0 for _ in range(len(A))] for _ in range(len(A))]
    U = [[0 for _ in range(len(A))] for _ in range(len(A))]
    for i in range(len(A)):
        L[i][i] = 1
        for j in range(i, len(A)):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, len(A)):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    return L, U


def get_inverse_matrix(A):
    L, U = LU_decomposition(A)
    L_inv = [[0 for _ in range(len(L))] for _ in range(len(L))]
    U_inv = [[0 for _ in range(len(U))] for _ in range(len(U))]
    for i in range(len(L)):
        L_inv[i][i] = 1 / L[i][i]
        for j in range(i + 1, len(L)):
            L_inv[j][i] = -sum(L[j][k] * L_inv[k][i] for k in range(i)) / L[i][i]
    for i in range(len(U)):
        U_inv[i][i] = 1 / U[i][i]
        for j in range(i + 1, len(U)):
            U_inv[j][i] = -sum(U[j][k] * U_inv[k][i] for k in range(i)) / U[i][i]

    return multiply_matrix(U_inv, multiply_matrix(L_inv, [[1, 0], [0, 1]]))


def get_jacobian(f1, f2, x, y):
    return [[f1_diff(x, y)[0], f2_diff(x, y)[0]], [f1_diff(x, y)[1], f2_diff(x, y)[1]]]


def get_newton_raphson(f1, f2, x, y):
    try:
        jacobian = get_jacobian(f1, f2, x, y)
        inverse_jacobian = get_inverse_matrix(jacobian)
        new_x = (
            x - inverse_jacobian[0][0] * f1(x, y) - inverse_jacobian[0][1] * f2(x, y)
        )
        new_y = (
            y - inverse_jacobian[1][0] * f1(x, y) - inverse_jacobian[1][1] * f2(x, y)
        )
        return [new_x, new_y]
    except:
        return [x + 0.1, y + 0.1]


real1, real2 = [2, 2], [-2, -2]
(x, y) = (0.5, 0.5)
epoch = 0
tolerance = 1e-4

while epoch < 100:
    (a, b) = get_newton_raphson(f1, f2, x, y)

    if (abs(a - real1[0]) < tolerance and abs(b - real1[1]) < tolerance) or (
        abs(a - real2[0]) < tolerance and abs(b - real2[1]) < tolerance
    ):
        print(f"수렴했습니다! 결과: x = {a:.4f}, y = {b:.4f}")
        break

    print(f"Epoch {epoch}:")
    print(
        f"real1 오차(%): x : {(real1[0]-a)/real1[0]:.4f}, y : {(real1[1]-b)/real1[1]:.4f}"
    )
    print(
        f"real2 오차(%): x : {(real2[0]-a)/real2[0]:.4f}, y : {(real2[1]-b)/real2[1]:.4f}"
    )
    print("---")

    x, y = a, b
    epoch += 1

if epoch == 100:
    print("100회 반복 후에도 수렴하지 않았습니다.")

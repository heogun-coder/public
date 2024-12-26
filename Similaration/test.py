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

    # 행렬식이 0에 가까운지 확인
    det = U[0][0] * U[1][1] - U[0][1] * U[1][0]
    if abs(det) < 1e-10:  # 행렬식이 너무 작으면
        raise ValueError("행렬식이 0에 가깝습니다")

    for i in range(len(L)):
        L_inv[i][i] = 1 / L[i][i]
        for j in range(i + 1, len(L)):
            L_inv[j][i] = -sum(L[j][k] * L_inv[k][i] for k in range(i)) / L[i][i]
    for i in range(len(U)):
        if abs(U[i][i]) < 1e-10:  # 대각 원소가 너무 작으면
            raise ValueError("대각 원소가 0에 가깝습니다")
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
        return [x, y]


import matplotlib.pyplot as plt

real1, real2 = [2, 2], [-2, -2]
(x, y) = (0, 1)
epoch = 0
tolerance = 1e-4

# 오차율을 저장할 리스트
errors_real1_x = []
errors_real1_y = []
errors_real2_x = []
errors_real2_y = []
epochs = []

while epoch < 100:
    (a, b) = get_newton_raphson(f1, f2, x, y)

    # 오차율 저장
    errors_real1_x.append((real1[0] - a) / real1[0])
    errors_real1_y.append((real1[1] - b) / real1[1])
    errors_real2_x.append((real2[0] - a) / real2[0])
    errors_real2_y.append((real2[1] - b) / real2[1])
    epochs.append(epoch)

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

# 그래프 그리기
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, errors_real1_x, "r-", label="x error")
plt.plot(epochs, errors_real1_y, "b-", label="y error")
plt.title("(2,2) error rate")
plt.xlabel("Epoch")
plt.ylabel("Error Rate")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, errors_real2_x, "r-", label="x error")
plt.plot(epochs, errors_real2_y, "b-", label="y error")
plt.title("(-2,-2) error rate")
plt.xlabel("Epoch")
plt.ylabel("Error Rate")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

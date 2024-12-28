def LU_decomposition(A):
    n = len(A)
    L = [[0 for _ in range(n)] for _ in range(n)]
    U = [[A[i][j] for j in range(n)] for i in range(n)]
    P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    # 대각선 요소를 1로 초기화
    for i in range(n):
        L[i][i] = 1

    # 각 열에 대해
    for i in range(n):
        # 피봇 선택 (i번째 열에서 가장 큰 절댓값을 가진 행 찾기)
        pivot = abs(U[i][i])
        pivot_row = i
        for j in range(i + 1, n):
            if abs(U[j][i]) > pivot:
                pivot = abs(U[j][i])
                pivot_row = j

        # 필요한 경우 행 교환
        if pivot_row != i:
            # U 행렬 교환
            U[i], U[pivot_row] = U[pivot_row], U[i]
            # P 행렬 교환
            P[i], P[pivot_row] = P[pivot_row], P[i]
            # L 행렬에서 이전 단계까지 계산된 부분 교환
            for k in range(i):
                L[i][k], L[pivot_row][k] = L[pivot_row][k], L[i][k]

        # 가우스 소거법 수행
        for j in range(i + 1, n):
            if U[i][i] != 0:  # 0으로 나누는 것 방지
                factor = U[j][i] / U[i][i]
                L[j][i] = factor
                for k in range(i, n):
                    U[j][k] -= factor * U[i][k]

    # 행렬식 계산
    determinant = 1
    for i in range(n):
        determinant *= U[i][i]

    return L, U, determinant, P


def get_inverse(matrix):
    L, U, _, P = LU_decomposition(matrix)
    n = len(matrix)
    inverse = [[0 for _ in range(n)] for _ in range(n)]

    # PA = LU이므로 A^(-1) = U^(-1)L^(-1)P 를 계산
    for k in range(n):
        # Step 1: Ly = Pb 풀기 (전진대입)
        y = [0 for _ in range(n)]
        b = [P[i][k] for i in range(n)]  # P의 k번째 열

        for i in range(n):
            sum = 0
            for j in range(i):
                sum += L[i][j] * y[j]
            y[i] = b[i] - sum

        # Step 2: Ux = y 풀기 (후진대입)
        x = [0 for _ in range(n)]
        for i in range(n - 1, -1, -1):
            sum = 0
            for j in range(i + 1, n):
                sum += U[i][j] * x[j]
            if U[i][i] != 0:  # 0으로 나누는 것 방지
                x[i] = (y[i] - sum) / U[i][i]

        # k번째 열에 결과 저장
        for i in range(n):
            inverse[i][k] = x[i]

    for x in range(len(inverse)):
        for y in range(len(inverse[x])):
            inverse[x][y] = round(inverse[x][y], 2)
    return inverse


def multiplication(A, B):
    result = [[0 for _ in range(len(A))] for _ in range(len(A))]
    for x in range(len(A)):
        for y in range(len(A[x])):
            for z in range(len(A[x])):
                result[x][y] += A[x][z] * B[z][y]
            # 소수점 둘째 자리 이하 반올림
            result[x][y] = round(result[x][y], 2)
    return result

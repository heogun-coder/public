import numpy as np
import matplotlib.pyplot as plt


def f(x):  # 함수 예
    return x**4 - 4 * x**3 + 3 * x**2 + 6


def numerical_derivative(func, x, epsilon=1e-4):  # 미분
    return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)


def gradient_descent(start, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    x = start  # 경사하강법
    for _ in range(max_iterations):
        grad = numerical_derivative(f, x)
        if abs(grad) < tolerance:
            break
        x -= learning_rate * grad
    return x, f(x)


def custom_optimization_sampling(start, search_range=2, num_samples=500):
    x_p = start  # 내 방법
    y_p = f(x_p)
    min_point = (x_p, y_p)  # 최소점 정의

    for angle in np.linspace(-89, 89, 180):
        slope = np.tan(np.radians(angle))  # 기울기 정의

        def rotated_line(x):
            return slope * (x - x_p) + y_p

        x_samples = np.linspace(x_p - search_range, x_p + search_range, num_samples)
        y_differences = np.abs(f(x_samples) - rotated_line(x_samples))

        idx = np.argmin(y_differences)
        x_candidate = x_samples[idx]

        if np.isclose(x_candidate, x_p, atol=1e-5):
            continue

        if f(x_candidate) < min_point[1]:
            min_point = (x_candidate, f(x_candidate))

    return min_point


start_point = np.random.uniform(-1, 3)

gd_min_point = gradient_descent(start_point)

custom_min_point = custom_optimization_sampling(start_point)

x_values = np.linspace(-1, 3, 300)
y_values = f(x_values)

plt.figure(figsize=(12, 6))
plt.plot(x_values, y_values, label="f(x) = x^4 - 4x^3 + 3x^2 + 6", color="blue")
plt.scatter(
    start_point, f(start_point), color="red", label="Starting Point", s=50, marker="o"
)
plt.scatter(
    gd_min_point[0],
    gd_min_point[1],
    color="green",
    label="Gradient Descent Minimum",
    s=50,
    marker="x",
)
plt.scatter(
    custom_min_point[0],
    custom_min_point[1],
    color="purple",
    label="Custom Optimization Minimum",
    s=50,
    marker="x",
)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Comparison of Gradient Descent and Custom Optimization")
plt.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt


# Define the function
def f(x):
    return x**4 - 4 * x**3 + 3 * x**2 + 6


# Numerical derivative calculation
def numerical_derivative(func, x, epsilon=1e-4):
    return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)


# Gradient Descent Implementation
def gradient_descent(start, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    x = start
    for _ in range(max_iterations):
        grad = numerical_derivative(f, x)
        if abs(grad) < tolerance:
            break
        x -= learning_rate * grad
    return x, f(x)


# Custom optimization implementation using increasing slopes
def custom_optimization_sampling(start, search_range=2, num_samples=500):
    x_p = start
    y_p = f(x_p)
    min_point = (x_p, y_p)  # Track the minimum point
    lines = []  # Store lines for plotting

    for slope in np.linspace(-10, 10, 200):  # Increment slope values gradually

        # Define the rotated line
        def rotated_line(x):
            return slope * (x - x_p) + y_p

        # Sampling x values around the starting point
        x_samples = np.linspace(x_p - search_range, x_p + search_range, num_samples)
        y_differences = np.abs(f(x_samples) - rotated_line(x_samples))

        # Find the index of the closest point on the graph
        idx = np.argmin(y_differences)
        x_candidate = x_samples[idx]

        # Avoid selecting the same point
        if np.isclose(x_candidate, x_p, atol=1e-5):
            continue

        # Update minimum point if a lower y value is found
        if f(x_candidate) < min_point[1]:
            min_point = (x_candidate, f(x_candidate))

        # Save line data for plotting
        x_line = np.linspace(x_p - search_range, x_p + search_range, 300)
        y_line = rotated_line(x_line)
        lines.append((x_line, y_line))

    return min_point, lines


# Settings for the algorithm
start_point = np.random.uniform(-1, 3)  # Random starting point

# Run the gradient descent algorithm
gd_min_point = gradient_descent(start_point)

# Run the custom optimization algorithm
custom_min_point, lines = custom_optimization_sampling(start_point)

# Generate values for plotting
x_values = np.linspace(-1, 3, 300)
y_values = f(x_values)

# Plot results
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
    marker="*",
)

# Plot rotated lines
for x_line, y_line in lines:
    plt.plot(x_line, y_line, color="orange", alpha=0.3)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Comparison of Gradient Descent and Custom Optimization")
plt.legend()
plt.grid()
plt.show()

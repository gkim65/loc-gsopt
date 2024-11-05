import numpy as np
from scipy.optimize import minimize

# Define the objective function (Rosenbrock function)
def rosenbrock(coords, a=1, b=100):
    x, y = coords
    return (a - x)**2 + b * (y - x**2)**2

# Initial guess for the coordinates (x, y)
initial_guess = np.array([0, 0])

# Perform the optimization using Nelder-Mead
result = minimize(rosenbrock, initial_guess, method='Nelder-Mead')

# Print the results
print("Optimal coordinates:", result.x)
print("Function value at optimal point:", result.fun)
print("Number of iterations:", result.nit)
print("Optimization successful:", result.success)

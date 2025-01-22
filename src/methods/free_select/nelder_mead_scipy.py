
# List to store simplex points
simplex_history = []

def callback(xk, convergence=None):
    simplex_history.append(xk.copy())

# Initial guess for the coordinates (x, y)
initial_guess = np.array([-30, -60])

# Perform the optimization using Nelder-Mead
result = minimize(cost_func_gap, 
                  initial_guess, 
                  method='Nelder-Mead',
                  options={'disp': True,
                          'maxiter': 5,
                           'initial_simplex' : np.array([[-30, -60],[-100, -60],[-30, -90]])},
                  callback=callback,
                  bounds = ((-180,180),(-90,90)))


# Coursera - Lagrange Polynomials
# https://yyelgalcvluedodqnnlkhn.coursera-apps.org/notebooks/W8_se_Lagrange_interpolation_simple.ipynb

# Importing libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Defining parameters
N = 5           # Order 5 polynomial chosen
x = np.linspace(-1, 1, 1000)
xi = [-1.0, -0.7650553239294647, -0.285231516480645, 0.285231516480645, 0.7650553239294647, 1.0]    # Weights taken

lag_poly = 1    # Initializing lagrange polynomial as 1

# Plotting the polynomials
plt.figure()
for i in range(-1, N):
    for j in range(-1, N):
        if j != i:
            lag_poly = lag_poly * ((x - xi[j + 1]) / float(xi[i + 1] - xi[j + 1]))

    plt.plot(x, lag_poly)

plt.ylim(-0.3, 1.1)
plt.title("Lagrange Polynomials of order %i" % N)
plt.show()

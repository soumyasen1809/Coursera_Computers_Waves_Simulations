# Coursera - Static Elasticity FEM
# https://yyelgalcvluedodqnnlkhn.coursera-apps.org/notebooks/W6_fe_static_elasticity.ipynb

# Importing libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Defining the parameters
x_n_points = 20                         # Number of points in X
x = np.linspace(0, 1, x_n_points)       # Define X co-ordinate
ele_len = x[2] - x[1]                   # Length of an element, assumed constant

f = np.zeros(x_n_points)                # Force vector
f[3*x_n_points/4] = 1                   # Assuming a force applied at 3/4th length of the domain
f[1] = 3                                # Force boundary condition
f[x_n_points - 2] = 1

u = np.zeros(x_n_points)                # Displacement vector
u[0] = 0.15                             # Boundary condition at x = 0
u[x_n_points - 1] = 0.05                # Boundary condition at x = L

mu = 1                                  # Shear modulus assumed constant

K = np.zeros((x_n_points, x_n_points))  # Defining a stiffness matrix
for i in range(1, x_n_points - 1):      # Stiffness matrix values directly used from lecture
    for j in range(1, x_n_points - 1):
        if i == j:
            K[i, j] = 2* mu/float(ele_len)
        elif i == j + 1:
            K[i, j] = -1 * mu/float(ele_len)
        elif i + 1 == j:
            K[i, j] = -1 * mu/float(ele_len)
        else:
            K[i, j] = 0

# Solving the matrix vector multiplication using np.dot()
u[1:x_n_points - 1] = np.dot(np.linalg.inv(K[1: x_n_points - 1, 1: x_n_points - 1]), np.transpose(f[1: x_n_points - 1]))

print ("The solution vector is ", u)

# Plotting the solution
plt.figure()
plt.plot(x, u, color='r', lw=1.5, label='Finite elements')
plt.title('Static Elasticity', size=16)
plt.ylabel('Displacement $u(x)$', size=16)
plt.xlabel('Position $x$', size=16)
plt.axis([0, 1, 0.04, .28])
plt.legend()
plt.grid(True)
plt.show()

# Coursera - W2V7 - Finite Difference Method - Higher Order
# https://yyelgalcvluedodqnnlkhn.coursera-apps.org/notebooks/W2_P2.ipynb


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import math


# Plotting a Gaussian function
x_max = 10                          # define a domain
n_points = 301                      # number of points
dx = x_max/float(n_points - 1)      # length of an finite element
a = 0.25
x0 = x_max/2                        # The function is symmetric about the line y = x0

x = np.linspace(0, x_max, n_points - 1)       # function f(x) defined
# Initialization of Gaussian function
f_x =(1./math.sqrt(2*np.pi*a))*np.exp(-(((x-x0)**2)/(float(2*a))))


# Plotting of gaussian
plt.plot(x, f_x)
plt.title('Gaussian function')
plt.xlabel('x, m')
plt.ylabel('Amplitude')
plt.xlim((0, x_max))
plt.grid()
plt.show()


# Comparing the 3 point operator second derivative numerical method with analytical method
num_der = np.linspace(0, x_max, n_points - 1)   # Defining the numerical solution array
ana_der = np.linspace(0, x_max, n_points - 1)   # # Defining the analytical solution array

for i in range(1,n_points - 3):
    num_der[i] = (f_x[i+1] - 2*f_x[i] + f_x[i-1])/(float(dx**2))

ana_der = 1./math.sqrt(2*np.pi*a)*((x-x0)**2/float(a**2) -1/float(a))*np.exp(-1/float((2*a))*(x-x0)**2)

ana_der[0] = 0                               # Excluding the boundary points
num_der[0] = 0
ana_der[n_points - 2] = 0
num_der[n_points - 2] = 0
num_der[n_points - 3] = 0

rms_error = math.sqrt(np.mean((num_der-ana_der)**2))


plt.plot (x, num_der,label="Numerical Derivative, 3 points", lw=2, color="violet")
plt.plot (x, ana_der, label="Analytical Derivative", lw=2, marker='+')
plt.plot (x, num_der-ana_der, label="Difference", lw=2, ls=":")
plt.title("Second derivative, Err (rms) = %.6f " % (rms_error) )
plt.xlabel('x, m')
plt.ylabel('Amplitude')
plt.legend(loc='lower left')
plt.grid()
plt.show()


# Comparing the 4 point operator second derivative numerical method with analytical method
num_der_4 = np.linspace(0, x_max, n_points - 1)   # Defining the numerical solution array

for i in range(1,n_points - 4):
    num_der_4[i] = (-1/float(12) * f_x[i - 2] + 4/float(3) * f_x[i - 1] - 5/float(2) * f_x[i] + 4/float(3) * f_x[i + 1] - 1/float(12) * f_x[i + 2]) / float(dx ** 2)

ana_der[0] = 0                               # Excluding the boundary points
num_der_4[0] = 0
ana_der[n_points - 2] = 0
num_der_4[n_points - 2] = 0
num_der_4[n_points - 3] = 0
num_der_4[n_points - 4] = 0

rms_error_2 = math.sqrt(np.mean((num_der_4-ana_der)**2))


plt.plot (x, num_der_4,label="Numerical Derivative, 4 points", lw=2, color="green")
plt.plot (x, ana_der, label="Analytical Derivative", lw=2, marker='+')
plt.plot (x, num_der_4-ana_der, label="Difference", lw=2, ls=":")
plt.title("Second derivative, Err (rms) = %.6f " % rms_error_2)
plt.xlabel('x, m')
plt.ylabel('Amplitude')
plt.legend(loc='lower left')
plt.grid()
plt.show()

# Printing the error for both the 3 and 4 point methods
print("RMS for 3 points is: ", rms_error)
print("RMS for 4 points is: ", rms_error_2)
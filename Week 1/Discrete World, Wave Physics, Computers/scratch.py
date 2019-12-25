import numpy as np;
import matplotlib.pyplot as plt;

sin_val = []
time = []

index = 0;

while index < 4*np.pi:
    val = np.sin(index);
    
    sin_val.append(val);
    time.append(index);

    index = index + 0.1;


plt.plot(time, sin_val)
plt.show()

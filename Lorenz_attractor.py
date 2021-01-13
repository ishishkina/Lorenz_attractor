import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

# definition of function that gives derivatives at the point x, y, z
def diff(x, y, z, sig=1.5, r=100, b=0.1):
    x_dot = sig * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


# time step
dt = 0.01
# number of time steps
count = 10**5
time = dt*count
t = np.linspace(0, time, count+1, endpoint=False)

# initialization of arrays
xs = np.empty(count + 1)
ys = np.empty(count + 1)
zs = np.empty(count + 1)

# set initial values
xs[0] = 0.01
ys[0] = 1.0
zs[0] = 1.05

# explicit Euler method
for i in range(count):
    # calculate derivatives at the current point
    x_dot, y_dot, z_dot = diff(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + x_dot * dt
    ys[i + 1] = ys[i] + y_dot * dt
    zs[i + 1] = zs[i] + z_dot * dt

# visualization
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(xs, ys, zs, lw=0.5, color="red")
ax. set_xlabel("X Axis")
ax. set_ylabel("Y Axis")
ax. set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()

# time dependence
fig = plt.figure()
ax = plt.axes()
ax. set_xlabel("t")
ax. set_ylabel("x")
plt.plot(t, xs, color="red")
plt.grid()
plt.show()

fig = plt.figure()
ax = plt.axes()
ax. set_xlabel("t")
ax. set_ylabel("y")
plt.plot(t, ys, color="blue")
plt.grid()
plt.show()

fig = plt.figure()
ax = plt.axes()
ax. set_xlabel("t")
ax. set_ylabel("z")
plt.plot(t, zs, color="green")
plt.grid()
plt.show()

# fast fourier transform
skip = 10000
xf = fftshift(fft(xs[skip:]))
frequency = fftshift(fftfreq(count - skip+1, dt))
fig = plt.figure()
ax = plt.axes()
ax. set_xlabel("frequency")
ax. set_ylabel("Xf")
plt.xlim(0.0, 2.0)
plt.plot(frequency, 1.0 / (count - skip) * np.abs(xf), color="blue")
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Create a mesh grid.
x = np.arange(0, 10, 1)
y = np.arange(0, 10, 1)

x, y = np.meshgrid(x, y)

# Transfer them to cylindrical coordinates.
a = np.arctan(y/x)
r = np.sqrt(np.square(x) + np.square(y))

# Show results.
plt.subplot(221)
plt.scatter(x, y)
plt.title('Cartesian Coordinates')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(222)
plt.scatter(a, r, c='red')
plt.title('Cylindrical Coordinates')
plt.xlabel('a')
plt.ylabel('r')

# For proof of concept apply the same idea in inverse.
a = np.arange(0, 2*np.pi, 2*np.pi/12)
r = np.arange(0, 10, 1)

a, r = np.meshgrid(a, r)

# Transfer them to cylindrical coordinates.
x = r*np.cos(a)
y = r*np.sin(a)

# Show results.
plt.subplot(223)
plt.scatter(x, y)
plt.title('Cartesian Coordinates')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(224)
plt.scatter(a, r, c='red')
plt.title('Cylindrical Coordinates')
plt.xlabel('a')
plt.ylabel('r')

plt.show()

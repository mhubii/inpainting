import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import torch
import torch.nn.functional as f

# Create a mesh grid.
x = np.arange(0, 10, 1) - 5
y = np.arange(0, 10, 1) - 5

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

# Check whether interpolation works with pytorch.
rad = np.zeros((128, 256))
rad[:8, :] = 1
rad[64:72, :] = 1
rad[-8:128, :] = 1

# Reference interpolate to cartesian space.
radius = np.arange(rad.shape[0]) - 0.5*rad.shape[0]
alpha = np.deg2rad(np.arange(0, rad.shape[1], 1))

radius, alpha = np.meshgrid(radius, alpha)

radius = radius.flatten()
alpha = alpha.flatten()

src_x = 0.5*rad.shape[0] + radius*np.cos(alpha)
src_y = 0.5*rad.shape[0] + radius*np.sin(alpha)

# Interpolate to regular grid.
dst_x, dst_y = np.meshgrid(np.arange(rad.shape[0]), np.arange(rad.shape[0]))

dst_x = dst_x.flatten()
dst_y = dst_y.flatten()

car = scipy.interpolate.griddata(
    (src_x, src_y),
    rad.T.flatten(),
    (dst_x, dst_y),
    method='linear',
    fill_value=0.0).reshape(rad.shape[0], rad.shape[0])

plt.subplot(222)
plt.imshow(car)
plt.title('RT in Cartesian Coordinates\nSampled in Cartesian Space')
plt.xlabel('x')
plt.ylabel('y')

# Interpolate to cartesian space with pytorch.


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


# rad[:, :int(0.5*rad.shape[1])] = 1
rad = np.expand_dims(rad, 0)
rad = np.expand_dims(rad, 0)
rad = torch.from_numpy(rad).float()

# Rearrange radon transform.

rad_pos = rad[:, :, :int(0.5*rad.shape[2]), :]
rad_neg = rad[:, :, int(0.5*rad.shape[2]):, :]
# rad_neg = rad_neg.flip() may be implemented on the next merge.
rad_neg = flip(rad_neg, 3)

"""
rad_pos = rad_pos.squeeze().numpy()
rad_neg = rad_neg.squeeze().numpy()
rad = rad.squeeze().numpy()

plt.subplot(221)
plt.title('pos')
plt.imshow(rad_pos)

plt.subplot(222)
plt.title('neg')
plt.imshow(rad_neg)

plt.subplot(223)
plt.title('whole')
plt.imshow(rad)

plt.show()
"""

#rad = torch.cat((rad_neg, rad_pos), 3)


rad = torch.cat((rad_pos[:, :, :, :], flip(rad_neg[:, :, :, :360-rad_pos.shape[3]], 2)), 3)

plt.subplot(221)
plt.imshow(rad.squeeze().numpy())
plt.title('RT in Cylindrical Coordinates')
plt.xlabel('a')
plt.ylabel('r')


x = np.arange(2*rad.shape[2]) - rad.shape[2]
y = np.arange(2*rad.shape[2]) - rad.shape[2]

x, y = np.meshgrid(x, y)

# The grid we want to sample the radon transform at.
r = np.sqrt(np.square(x) + np.square(y))
a = np.where(r == 0, 0, np.where(x >= 0, np.arcsin(y/r), -np.arcsin(y/r) + np.pi))

# Normalize.
r = r/(0.5**2*rad.shape[2]*np.sqrt(2)) - 1
a = np.rad2deg(a)/180 - 1

grid = np.stack((a, r), axis=2)
grid = np.expand_dims(grid, 0)

grid = torch.from_numpy(grid).float()

car = f.grid_sample(rad, grid, padding_mode='zeros')

car = car.squeeze()
car = car.numpy()

plt.subplot(223)
plt.imshow(car)
plt.title('RT in Cartesian Coordinates\nSampled in Cylindrical Space')
plt.xlabel('x')
plt.ylabel('y')

plt.show()

"""
plt.subplot(121)
plt.scatter(a, r, c='red')
plt.title('Cylindrical')

plt.subplot(122)
plt.scatter(x, y)
plt.title('Cartesian')

plt.show()
"""

import numpy as np
from skimage.draw import ellipse
import scipy.misc


class Create2DRandVol:
    """
        Create2DRandVol serves to simulate 2D random volumes.

    """
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.vol = np.zeros([nx, ny])

    def ellipses(self, nmat, nell, seed=None):
        """
            Fills the volume with random ellipses of nmat different materials
            and nell ellipses of this material. A seed can be set to create
            reproducible results.

        """

        self.vol = np.zeros([self.nx, self.ny])

        if seed:
            np.random.seed(seed)

        shape = self.vol.shape
        mu = np.random.rand(nmat)

        for n in range(nmat):
            for _ in range(nell):
                # Center of the ellipse.
                r = np.random.randint(0, self.nx)
                c = np.random.randint(0, self.ny)

                # Minor and major semi-axis.
                r_rad = np.random.randint(1, int(self.nx*0.5))
                c_rad = np.random.randint(1, int(self.ny*0.5))

                # Rotation.
                rot = np.random.rand()*2*np.pi

                rr, cc = ellipse(r=r, c=c, r_radius=r_rad, c_radius=c_rad,
                                 rotation=rot, shape=shape)

                self.vol[rr, cc] = mu[n]

        return self.vol


if __name__ == '__main__':
    # Set some initial parameters.
    nx = 64
    ny = 64
    nVol = 1000

    # Create random volume of ellipses.
    vol = Create2DRandVol(nx, ny)
    data = []

    for _ in range(nVol):
        nMat = np.random.randint(1, 5)
        nEll = np.random.randint(1, 2)

        data.append(vol.ellipses(nMat, nEll))

    data = np.asarray(data).reshape([nVol, nx, ny])

    # Save data.
    np.save('../data/rand_vol_1k', data)
    scipy.misc.imsave('../img/rand_ell.png', data[0])


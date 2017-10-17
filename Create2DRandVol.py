"""
    Doc 2017.09.29

    Create2DRandVol serves to simulate 2D random volumes that shall serve
    as training and test data.

    ellipses(nmat, nell, seed):

        Fills the volume with random ellipses of nmat different materials
        and nell ellipses of this material.

"""
import numpy as np
from skimage.draw import ellipse


class Create2DRandVol:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.vol = np.zeros([nx, ny])

    def ellipses(self, nmat, nell, seed=None):
        self.vol = np.zeros([self.nx, self.ny])

        if seed:
            np.random.seed(seed)

        shape = self.vol.shape
        mu = np.random.rand(nmat)

        for n in range(nmat):
            for _ in range(nell):
                # center of the ellipse
                r = np.random.randint(0, self.nx)
                c = np.random.randint(0, self.ny)

                # minor and major semi-axis
                r_rad = np.random.randint(1, int(self.nx*0.5))
                c_rad = np.random.randint(1, int(self.ny*0.5))

                # rotation
                rot = np.random.rand()*2*np.pi

                rr, cc = ellipse(r=r, c=c, r_radius=r_rad, c_radius=c_rad,
                                 rotation=rot, shape=shape)

                self.vol[rr, cc] = mu[n]

        return self.vol


if __name__ == '__main__':
    # set some initial parameters
    nx = 128
    ny = 128
    nImg = 1000

    # create random volume of ellipses
    vol = Create2DRandVol(nx, ny)
    data = []

    for _ in range(nImg):
        nMat = np.random.randint(1, 5)
        nEll = np.random.randint(1, 2)

        data.append(vol.ellipses(nMat, nEll))

    data = np.asarray(data).reshape([nImg, nx, ny])

    # save data
    np.save('randVol1k', data)


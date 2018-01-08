import numpy as np
from skimage.draw import ellipse
import scipy.misc
import argparse
import utils


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
    parser = argparse.ArgumentParser(description='Creates random 2d volume.')
    parser.add_argument('-n_mat', type=int, default=5,
                        help='Number of randomly initialized materials (default = 5).')
    parser.add_argument('-n_ell', type=int, default=3,
                        help='Number of randomly initialized ellipses of one material (default = 3).')

    args = parser.parse_args()

    # Create random volume of ellipses.
    vol = Create2DRandVol(utils.N_X, utils.N_Y)
    data = np.empty([utils.N_VOL, utils.N_X, utils.N_Y])

    for n in range(utils.N_VOL):
        data[n] = vol.ellipses(args.n_mat, args.n_ell)

    # Save data.
    np.save('../data/rand_vol_{}_test'.format(utils.N_VOL), data)
    scipy.misc.imsave('../img/rand_ell.png', data[0])


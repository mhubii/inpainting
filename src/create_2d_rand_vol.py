import numpy as np
from skimage.draw import ellipse
from tqdm import tqdm
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

    # Stance circular mask.
    def stance_circular_mask(self):
        """
            Creates a circular mask centered in the center
            of the 2d volume and applies it to the volume.

        """

        # Center of the volume.
        center = [int(self.ny / 2), int(self.nx / 2)]

        y, x = np.ogrid[:self.nx, :self.ny]
        dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        mask = dist_from_center <= min(self.nx, self.ny)*0.5

        self.vol = np.multiply(self.vol, mask)

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

    for n in tqdm(range(utils.N_VOL)):
        vol.ellipses(args.n_mat, args.n_ell)
        data[n] = vol.stance_circular_mask()

    # Save data.
    np.save('../data/rand_vol_{}'.format(utils.N_VOL), data)
    # scipy.misc.imsave('../img/rand_ell.png', data[0])


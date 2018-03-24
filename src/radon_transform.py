import numpy as np
from skimage.transform import radon
import scipy.misc
from tqdm import tqdm
import os
import csv
import utils


# Radon transform.
def radon_transform():
    """
        Takes randomly generated 2d volumes and performs
        a radon transformation on them.

    """

    # Load data.
    data = np.load('../data/rand_vol_{}.npy'.format(utils.N_VOL))
    data.reshape([utils.N_VOL, utils.N_X, utils.N_Y])

    # Perform the radon transformation.
    angles = np.linspace(0., utils.A, utils.N_A, endpoint=False)
    rad = np.empty([utils.N_VOL, utils.N_X, utils.N_A])

    for n in tqdm(range(utils.N_VOL)):
        rad[n] = radon(data[n], angles, circle=True)

    data = np.asarray(rad).reshape([utils.N_VOL, rad[0].shape[0], rad[0].shape[1]])

    return data


# Single radon transform.
def single_radon_transform(n):
    """
        Takes randomly generated 2d volumes and performs
        a radon transformation on one volume.

    """

    # Load data.
    data = np.load('../data/rand_vol_{}.npy'.format(utils.N_VOL))
    data.reshape([utils.N_VOL, utils.N_X, utils.N_Y])

    # Perform the radon transformation.
    angles = np.linspace(0., utils.A, utils.N_A, endpoint=False)

    rad = radon(data[n], angles, circle=True)

    data = np.asarray(rad).reshape([rad.shape[0], rad.shape[1]])

    scipy.misc.imsave('../img/rand_ell_rad.png', data)

# Save.
def save():
    """
        Saves single images and writes their locations
        to a .csv file.

    """

    # Save data in snippets and store locations as .csv file.
    loc = list()

    for idx, vol in enumerate(data):
        scipy.misc.imsave('../data/radon_transforms/radon_n_vol_{}.jpg'.format(idx), vol)
        loc.append([os.path.abspath(os.path.join(os.getcwd(), '../data/radon_transforms/radon_n_vol_{}.jpg'.format(idx)))])

    with open('../data/locations', 'w') as my_file:
        wr = csv.writer(my_file)
        wr.writerows(loc)


# Save as snippets.
def save_as_snippets():
    """
        Turns an image into snippets and stores them and
        their location in a .csv file.

    """

    # Save data in snippets and store locations as .csv file.
    loc = list()

    for n in tqdm(range(utils.N_VOL)):
        sliced_data, _, _ = utils.slice_array(data[n], utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH,
                                              (utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH))

        for idx, slice in enumerate(sliced_data):
            scipy.misc.imsave('../data/radon_transform_snippets/radon_n_vol_{}_idx_{}.jpg'.format(n, idx), slice)
            loc.append([os.path.abspath(os.path.join(os.getcwd(), '../data/radon_transform_snippets/radon_n_vol_{}_idx_{}.jpg'.format(n, idx)))])

    with open('../data/locations_snippets', 'w') as my_file:
        wr = csv.writer(my_file)
        wr.writerows(loc)


if __name__ == '__main__':
    # Create radon transforms.
    # data = radon_transform()
    single_radon_transform(6010)

    # Save results.
    # save()
    save_as_snippets()

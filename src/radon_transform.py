import numpy as np
from skimage.transform import radon
import scipy.misc
import os
import csv
import utils


def slice_array(arr,
                n_rows, n_cols):
    """
        Slices an array into multiple sub-arrays of size
        n_rows x n_cols and returns them as a list.

    """

    sliced_arr = list()

    # Determine how many sub-arrays can be obtained.
    n_row_slice = int(arr.shape[0]/n_rows)
    n_col_slice = int(arr.shape[1]/n_cols)

    for i in range(n_row_slice):
        for j in range(n_col_slice):
            sliced_arr.append(arr[i*n_rows:(i + 1)*n_rows, j*n_cols:(j + 1)*n_cols])

    return sliced_arr


if __name__ == '__main__':
    # Load data.
    data = np.load('../data/rand_vol_{}.npy'.format(utils.N_VOL))
    data.reshape([utils.N_VOL, utils.N_X, utils.N_Y])

    # Perform the radon transformation.
    angles = np.linspace(0., utils.A, utils.N_A, endpoint=False)
    rad = np.empty([utils.N_VOL, utils.N_X, utils.N_A])

    for n in range(utils.N_VOL):
        rad[n] = radon(data[n], angles, circle=True)

    data = np.asarray(rad).reshape([utils.N_VOL, rad[0].shape[0], rad[0].shape[1]])

    # Save data in snippets and store locations as .csv file.
    loc = list()

    for n in range(utils.N_VOL):
        sliced_data = slice_array(data[n], utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH)

        for idx, slice in enumerate(sliced_data):
            scipy.misc.imsave('../data/radon_transform_snippets/radon_n_vol_{}_idx_{}.jpg'.format(n, idx), slice)
            loc.append([os.path.abspath(os.path.join(os.getcwd(), '../data/radon_transform_snippets/radon_n_vol_{}_idx_{}.jpg'.format(n, idx)))])

    with open('../data/locations', 'w') as my_file:
        wr = csv.writer(my_file)
        wr.writerows(loc)

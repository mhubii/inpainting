import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import argparse


class SinoNet:
    """
    Doc - 2017.10.31

        SinoNet may perform the inpainting of a sinogram by using
        a deep convolutional generative adversarial network (DCGAN).

        The implemented DCGAN follows the paper of Alec Radford,
        Luke Metz, and Soumith Chintala:

            arXiv: https://arxiv.org/abs/1511.06434

        The implemented inpainting follows the paper of
        Raymond A. Yeh et al.:

            arXiv: https://arxiv.org/abs/1607.07539

        Code by parts from Yash Katariya:

            GitHub: https://github.com/yashk2810/DCGAN-Keras

    """

    def __init__(self):
        pass

    def build_model(self):
        # generate the models
        self.d = self.discriminator()
        self.g = self.generator()
        self.d_g = self.discriminate_generator(self.g, self.d)

        # set losses and optimizers
        self.d.compile(loss='binary_crossentropy', optimizer=Adam())
        self.g.compile(loss='binary_crossentropy', optimizer=Adam())
        self.d_g.compile(loss='binary_crossentropy', optimizer=Adam())

    def train_model(self, epoch=10, batch_size=128):
        batch_count = int(self.X_train.shape[0] / batch_size)

        for i in range(epoch):
            for j in range(batch_count):
                # input for generator
                noise_input = np.random.rand(batch_size, 100)

                # getting random images from X_train of size=batch_size
                # these are the real images that will be fed to the discriminator
                image_batch = self.X_train[np.random.randint(0, self.X_train.shape[0], size=batch_size)]

                # these are the predicted images from the generator
                predictions = self.g.predict(noise_input, batch_size=batch_size)

                # the discriminator takes in the real images and the generated images
                X = np.concatenate([predictions, image_batch])

                # labels for the discriminator
                y_discriminator = [0] * batch_size + [1] * batch_size

                # train the discriminator
                self.d.trainable = True
                self.d.train_on_batch(X, y_discriminator)

                # train the generator
                noise_input = np.random.rand(batch_size, 100)
                y_generator = [1] * batch_size
                self.d.trainable = False
                self.d_g.train_on_batch(noise_input, y_generator)

        # save results
        self.d.save_weights('TrainedNets/discriminator.h5')
        self.g.save_weights('TrainedNets/generator.h5')

    def complete(self):
        pass

    def generate(self):
        self.g.load_weights('TrainedNets/generator.h5')

        # generate random images
        try_input = np.random.rand(100, 100)
        gen = self.g.predict(try_input)

        # save the results
        np.save(gen, 'Data/genSin')

    def load_data(self, loc):
        # load the images
        self.X_all = np.load(loc)
        self.X_all.reshape([10000, 64, 64])

        # scale the range of the image to [-1, 1]
        # because we are using tanh in the last layer of the generator
        self.X_train = (self.X_train - 127.5) / 127.5

    def generator(self):
        model = Sequential()
        model.add(Dense(input_dim=100, units=32768, activation='relu'))
        model.add(Reshape((8, 8, 512)))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=256, kernel_size=(5, 5), padding='same', strides=(2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=128, kernel_size=(5, 5), padding='same', strides=(2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(filters=1, kernel_size=(5, 5), padding='same', strides=(2, 2), activation='tanh'))
        return model

    def discriminator(self):
        model = Sequential()
        model.add(Conv2D(input_shape=(64, 64, 1), filters=128, kernel_size=(5, 5), padding='same', strides=(2, 2)))
        model.add(LeakyReLU())
        model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same', strides=(2, 2)))
        model.add(LeakyReLU())
        model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same', strides=(2, 2)))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(units=1, activation='sigmoid'))

        return model

    def discriminate_generator(self, g, d):
        model = Sequential()
        model.add(g)
        d.trainable = False
        model.add(d)
        return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--path', type=str, default='Data/radonRandVol1kGroundTruth.npy')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    # setup the model
    gan = SinoNet()
    gan.build_model()
    gan.load_data(args.path)

    # train or generate
    if args.mode == 'train':
        gan.train_model()
    elif args.mode == 'generate':
        gan.generate()

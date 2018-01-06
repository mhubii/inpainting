import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from scipy.signal import convolve2d


"""
    The implementation of this algorithm is based on the paper
    of Raymond A. Yeh et al.:
    
        arXiv: https://arxiv.org/pdf/1607.07539.pdf 
        

"""


class Inpainting:
    def __init__(self, generator, discriminator):
        self.gen = generator
        self.dis = discriminator

    def inpaint(self, img, mask, blend=True):
        pass

    def backpropagation(self):
        return 0

    def context_loss(self, img, mask):
        pass

    def prior_loss(self):
        pass



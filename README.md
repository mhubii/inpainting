# inpainting

Goal of inpainting is to complete **sinograms** (radon transforms) as acquired from **computer tomographic** (CT) imaging using a **deep convolutional adversarial neural net** (DCGAN). The desired sinograms need to be completed since we aim at reducing the number of projections and therefore to effectively **reduce the dose** from x-rays that a patient would be exposed to.

The concept needs to be proven before applying it to clinical data. Therefore, a simple application **Create2DRandVol.py** simulates a 2D volume of randomly generated ellipses to create training data.

Once the data is acquired, may it be clinical or simulated, it is fed into **SinoNet.py**, which holds the implementation of the DCGAN and may complete the incomplete data.

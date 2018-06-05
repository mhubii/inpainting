# Inpainting

## Introduction

Our goal is to perform an inpainting for computer tomographic (CT) images, to reduce the dose a patient has to undergo during an examination. We take advantage of recent developments in the architecture of neural nets and utilize so called deep convolutional generative adversarial neural nets (DCGANs).

### Radon Transforms

Radon transforms are images as one obtains them from CT scans before the reconstruction. Figure 1 shows a reconstructed CT scan in the axial plane of a patient.

<br>
<figure>
  <p align="center"><img src="img/real_ct.png" width="20%" height="20%"></p>
  <figcpation>Fig. 1: Reconstructed CT Scan of a Liver. [<a href="#1">1</a>]</figcaption>
</figure>
<br><br>

Before the reconstruction, an image rather has the shape of many stacked sine functions. This typical appearance, as seen below, results from the detector and the x-ray source which rotate around the patient. 

<br>
<figure>
  <p align="center"><img src="img/rand_ell.png" width="20%" height="20%" hspace="40"><img src="img/rand_ell_rad.png"      width="20%" height="20%" hspace="40"></p>
  <figcaption>Fig. 2: (Left) Section through randomly simulated ellipses. (Right) Radon transform of these ellipses.</figcaption>
</figure>
<br><br>

The shown radon transform is the result of transforming a simulated section through randomly created ellipses. For a proof of concept we stick to these simulated sections before applying the algorithm to real patient data.

### Dose Reduction
In order to reduce the radiation, a patient has to undergo, one can simply reduce the number of projections. An example of a radon transform with a reduced number of projections is shown in figure 3.

<br>
<figure>
  <p align="center"><img src="img/rand_ell_rad_less_dose.png" width="20%" height="20%"></p>
  <figcaption>Fig. 3: Radon transform with less projections.</figcaption>
</figure>
<br><br>

To compensate for the reduced information that one obtains from such a radon transform, we employ a DCGAN for inpainting the unkown regions. Please note that due to aliasing the masked regions of the radon transform can be barely seen on a computer screen.

## Methods


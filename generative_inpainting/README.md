# Generative Inpainting
As proposed in [<a href="#1">1</a>], a DCGAN can be used for inpainting purposes. For the architecture of the DCGAN, we follow the recipe propsoed by [<a href="#2">2</a>]. Since this technique works  well for small image sizes but doesnt for highly resolved images, as in our case, we simply train the algorithm on smaller image snippets.

## Results
The DCGAN produces radon transform snippets that can be hardly distinguished from real radon transform snippets as shown below. The network learns to model the smooth transitions in a radon transform.

<br>
<figure>
  <p align="center"><img src="img/real_snippet.png" width="30%" height="30%"><img src="img/snippet_at_epoch_20.png" width="30%" height="30%"><img src="img/epochs.gif" width="30%" height="30%"></p>
  <figcaption>Fig. 1: (Left) Snippets of size 63x63 of a radon transform. (Center) Radon transform snippets as created by the generator of the DCGAN. (Right) Evolution of the generator.</figcaption>
</figure>
<br><br>

When the inpainting is applied on overlaying snippets of the radon transform, the resulting images do capture local structures quite well but not global structures, as can be seen in figure 2.

<br>
<figure>
  <p align="center"><img src="img/original.png" width="20%" height="20%" hspace="40"><img src="img/masked.png" width="20%" height="20%" hspace="40"><img src="img/optimal.png" width="20%" height="20%" hspace="40"></p>
  <figcaption>Fig. 2: (Left) Radon transform of randomly simulated ellipses, using 100% dose. (Center) Radon transform using only 25% dose. (Right) Inpainted radon transform. </figcaption>
</figure>
<br><br>

The reconstructed images (Figure 3) reveal that the inpainted radon transform results in blurry images and the missing of global consistency produces some ring artifacts.

<br>
<figure>
  <p align="center"><img src="img/reco_original.png" width="20%" height="20%" hspace="40"><img src="img/reco_masked.png" width="20%" height="20%" hspace="40"><img src="img/reco_optimal.png" width="20%" height="20%" hspace="40"></p>
  <figcaption>Fig. 3: (Left) Reconstruction of original radon transform. (Center) Reconstruction with less projections. (Right) Reconstruction using the inpainted radon transform. </figcaption>
</figure>
<br><br>

## Conclusion
The presented method, as it is, does not work well to perform inpaintings on radon transformations. Other techniques have to be investigated to take the the global structures into account. Further, more computational power and a bigger training set could increase the resultion of the images.

## Literature
[<a name="1">1</a>] [Semantic Image Inpainting with Deep Generative Models](https://arxiv.org/abs/1607.07539 "Link to arXiv")

[<a name="2">2</a>] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434 "Link to arXiv")

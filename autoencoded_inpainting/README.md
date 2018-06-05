# Autoencoded Inpainting

As proposed in [<a href="#1">1</a>] an autoencoder can be used for inpainting purposes. Key to this approach is the use of dilated convolutions in the bottleneck of the autoencoder (AE), to effectively increase the receptive field and to thus generate globally consistent inpaintings. Further, a local and global discriminator are used to enhance local and global consistency further.

## Results

The inpainting produces patterns that reflect local and global structures well, as can be seen in figure 2.

<br>
<figure>
  <p align="center"><img src="img/original.png" width="20%" height="20%" hspace="40"><img src="img/masked.png" width="20%" height="20%" hspace="40"><img src="img/inpainted.png" width="20%" height="20%" hspace="40"></p>
  <figcaption>Fig. 1: (Left) Radon transform of randomly simulated ellipses, using 100% dose. (Center) Radon transform using only 25% dose. (Right) Inpainted radon transform. </figcaption>
</figure>
<br><br>

The reconstruction reveals that the inpainting adds some blurring to the image, which is caused be the loss of information when using less projections.

<br>
<figure>
  <p align="center"><img src="img/original_reco.png" width="20%" height="20%" hspace="40"><img src="img/masked_reco.png" width="20%" height="20%" hspace="40"><img src="img/inpainted_reco.png" width="20%" height="20%" hspace="40"></p>
  <figcaption>Fig. 2: (Left) Reconstruction of original radon transform. (Center) Reconstruction with less projections. (Right) Reconstruction using the inpainted radon transform. </figcaption>
</figure>
<br><br>

## Conclusion
The autoencoded inpainting, presented in [<a href="#1">1</a>], is not quite as useful in the setting of radon transformations, since the masks differ a lot to from what the paper proposes. Though, the AE architecture with dilated convolutions seems to work well here but there is no benefit from using a global discriminator, since the receptive field of the AE already captures the whole coherence of the problem. The only lack that comes with this network architecture is the natural blurring of the reconstructed image that goes along with the loss of information (Figure 2, right). Key for evolving this technique in this setting will thus be the extended use of a network architecture with a more sophisticated local discriminator. Furthermore, one could simply apply a filter to the reconstruction or try a network that receives feedback from the sharpness of objects in the reconstruced image space. The technique shows promissing results and should be further investigated with the above listed possible improvements.

## Literature
[<a name="1">1</a>] [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf "Link to Their Website")

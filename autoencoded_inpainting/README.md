# Autoencoded Inpainting

As proposed in [<a href="#1">1</a>] an autoencoder can be used for inpainting purposes. 

## Results

<br>
<figure>
  <p align="center"><img src="img/original.png" width="20%" height="20%" hspace="40"><img src="img/masked.png" width="20%" height="20%" hspace="40"><img src="img/inpainted.png" width="20%" height="20%" hspace="40"></p>
  <figcaption>Fig. 1: (Left) Radon transform of randomly simulated ellipses, using 100% dose. (Center) Radon transform using only 25% dose. (Right) Inpainted radon transform. </figcaption>
</figure>
<br><br>

<br>
<figure>
  <p align="center"><img src="img/original_reco.png" width="20%" height="20%" hspace="40"><img src="img/masked_reco.png" width="20%" height="20%" hspace="40"><img src="img/inpainted_reco.png" width="20%" height="20%" hspace="40"></p>
  <figcaption>Fig. 2: (Left) Reconstruction of original radon transform. (Center) Reconstruction with less projections. (Right) Reconstruction using the inpainted radon transform. </figcaption>
</figure>
<br><br>

## Conclusion



## Literature
[<a name="1">1</a>] [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf "Link to Their Website")

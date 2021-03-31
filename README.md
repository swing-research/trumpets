Official repo for Trumpets [(paper)](https://arxiv.org/abs/2102.10461)

Trumpets are injective variants of normalizing flows that allow for a small latent space dimension compared to the dataset size. This greatly improves the speed of training (upto 10x faster) while being comparable in terms of sample quality.

Other than image generation, we are more interested in inference applications---MAP estimation for inverse problems and UQ. We findthat injectivity of Trumpets lead to much better performance than baselines given the same generator architecture.

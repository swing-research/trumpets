Official repo for Trumpets [(paper)](https://arxiv.org/abs/2102.10461)

Trumpets are injective variants of normalizing flows that allow for a small latent space dimension compared to the dataset size. This greatly improves the speed of training (upto 10x faster) while being comparable in terms of sample quality.

Other than image generation, we are more interested in inference applications---MAP estimation for inverse problems and UQ. We findthat injectivity of Trumpets lead to much better performance than baselines given the same generator architecture.

```
usage: train.py [-h] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--dataset DATASET] [--lr LR]
                [--ml_threshold ML_THRESHOLD] [--model_depth MODEL_DEPTH] [--latent_depth LATENT_DEPTH] [--learntop LEARNTOP]
                [--gpu_num GPU_NUM] [--remove_all REMOVE_ALL] [--desc DESC] [--train] [--notrain] [--inv] [--noinv] [--posterior]
                [--noposterior] [--calc_logdet] [--nocalc_logdet] [--inv_prob INV_PROB] [--snr SNR]
                [--inv_conv_activation INV_CONV_ACTIVATION] [--T T]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        number of epochs to train for
  --batch_size BATCH_SIZE
                        batch_size
  --dataset DATASET     which dataset to work with
  --lr LR               learning rate
  --ml_threshold ML_THRESHOLD
                        when should ml training begin
  --model_depth MODEL_DEPTH
                        revnet depth of model
  --latent_depth LATENT_DEPTH
                        revnet depth of latent model
  --learntop LEARNTOP   Trainable top
  --gpu_num GPU_NUM     GPU number
  --remove_all REMOVE_ALL
                        Remove the previous experiment
  --desc DESC           add a small descriptor to folder name
  --train
  --notrain
  --inv
  --noinv
  --posterior
  --noposterior
  --calc_logdet
  --nocalc_logdet
  --inv_prob INV_PROB   choose from denoising (default) | sr | randmask | randgauss
  --snr SNR             measurement SNR (dB)
  --inv_conv_activation INV_CONV_ACTIVATION
                        activation of invertible 1x1 conv layer
  --T T                 sampling tempreture

```

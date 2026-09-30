# Experiment 7: Autoencoders, Convolutional, Denoising and Variational Autoencoders on MNIST

## Aim

The objective of this experiment is to develop an end-to-end understanding of autoencoders and their variants for
image representation, reconstruction, denoising and generative modeling.
The experiment begins with a fully connected autoencoder, progresses to a Convolutional Autoencoder (CAE), intro
duces image corruption for denoising, and finally develops a Variational Autoencoder (VAE). Students will visualize
the learned latent representation, compare reconstruction quality using quantitative and qualitative measures, and
explore the generative capability of the VAE.

## Dataset Description

**MNIST** contains grayscale handwritten digits (0-9), each 28 x 28 x 1.
- **Training:** the first 10,000 training images, split into 9,000 for training and 1,000 for validation.
- **Testing:** the first 2,000 test images.
- **Preprocessing:** pixels are scaled from [0, 255] to [0, 1]. The FC model flattens images to 784 values, and the convolutional models keep the 28 x 28 x 1 shape.
- **Labels:** they are never used as training targets. They are used only to colour latent-space plots and to train a small auxiliary CNN classifier that judges how recognisable generated images are.

## Procedure

1. Load, normalise and split the data.
2. Train an FC autoencoder and plot originals, reconstructions, error maps and loss curves.
3. Train a CAE on the same data and compare it with the FC-AE.
4. Corrupt images with Gaussian and salt-and-pepper noise and train denoising CAEs against the clean targets.
5. Train a VAE with a 2-D latent space, then visualise the latent space, sample from the prior and interpolate between codes.
6. Compute per-image error distributions and analyse the five worst reconstructions.
7. Run the latent-dimension study (d_z = 2, 8, 16, 32).
8. Complete the eight additional exercises and answer the 25 discussion questions.

All autoencoders were trained for 20 epochs with Adam (learning rate 1e-3), batch size 128 and binary cross-entropy.

## Techniques Used

- Fully connected autoencoder (784-128-32-16-32-128-784)
- Convolutional autoencoder (Conv32, Pool, Conv64, Pool, Conv64, then upsampling back to 28 x 28)
- Denoising autoencoder (Gaussian and salt-and-pepper noise)
- Variational autoencoder with the reparameterisation trick
- Metrics: MSE, MAE and SSIM
- Latent-space visualisation, interpolation and per-image error analysis
- beta-VAE and transposed-convolution decoder variants

## Brief Descriptions

- **Autoencoder:** an encoder compresses the input to a low-dimensional latent code z, and a decoder reconstructs the input from it. The bottleneck forces the network to learn structure rather than copy pixels.
- **Convolutional AE:** it uses local, weight-shared filters and keeps the 2-D layout. The 7 x 7 x 64 latent map preserves neighbourhood relationships and needs fewer parameters.
- **Denoising AE:** it receives a corrupted image but is trained to output the clean one. Gaussian noise uses sigma in {0.1, 0.2, 0.3}, and salt-and-pepper noise uses p in {0.05, 0.10, 0.20}.
- **VAE:** the encoder outputs a mean and log-variance, and z = mu + sigma * epsilon with epsilon drawn from N(0, I). The loss is reconstruction (BCE) plus KL divergence to the N(0, I) prior, which makes the latent space continuous and sampleable.
- **MSE / MAE / SSIM:** MSE penalises large pixel errors, MAE is the average absolute deviation, and SSIM compares luminance, contrast and structure, which is closer to perceived quality.

## Results and Evaluation

| Model | MSE | MAE | SSIM | Params | Time (s) |
|---|---|---|---|---|---|
| FC Autoencoder | 0.02046 | 0.05514 | 0.75857 | 211,040 | 10.97 |
| Conv. Autoencoder | 0.00259 | 0.01500 | 0.97395 | 74,497 | 18.35 |
| Denoising CAE (sigma = 0.2) | 0.00448 | 0.02091 | 0.94638 | 74,497 | 18.32 |
| VAE (z = mu) | 0.04639 | 0.10963 | 0.44144 | 134,165 | 21.35 |

**VAE losses (per image):** reconstruction 155.78, KL 5.29, total 161.07.

**FC vs CAE:** the CAE cut MSE by about 87% (0.02046 to 0.00259) with 65% fewer parameters. Preserving spatial locality gives sharper, more faithful reconstructions.

**Latent dimension (FC-AE):**

| d_z | 2 | 8 | 16 | 32 |
|---|---|---|---|---|
| MSE | 0.04577 | 0.02663 | 0.02046 | 0.01892 |
| SSIM | 0.458 | 0.689 | 0.759 | 0.777 |

Quality improves with d_z, but gains diminish beyond 16.

**Denoising:** the denoised SSIM stays high while the noisy input degrades. Values below are for sigma = 0.1 / 0.2 / 0.3.

| | sigma = 0.1 | sigma = 0.2 | sigma = 0.3 |
|---|---|---|---|
| Denoised SSIM (Gaussian) | 0.961 | 0.946 | 0.915 |
| Noisy-input SSIM | 0.679 | 0.594 | 0.509 |

For salt-and-pepper noise (p = 0.05 / 0.10 / 0.20), denoised SSIM is 0.954 / 0.943 / 0.905. Each denoiser performs best on the noise type it was trained on. The Gaussian-trained model transfers reasonably to salt-and-pepper noise, but the reverse is much weaker.

**Error analysis:** FC-AE errors are unimodal and right-skewed, with a mean of 0.0205 and a 95th percentile of 0.0394. Four of the five worst images are the digit 8 (thick, complex strokes or unusual styles).

**VAE latent space:**
- Digits 0 and 1 separate clearly, while 2/3/5/6/8 and 4/7/9 overlap heavily.
- The 5-NN accuracy is 49.0% (FC-AE: 54.4%).
- Interpolation is smooth, whereas pixel blending only cross-fades images.
- Generated samples are blurry. Only 15.85% of prior samples decoded by the VAE were confidently recognised, against 4.75% for the FC-AE decoder.

## Summary
The convolutional autoencoder gave the best reconstructions (SSIM 0.974), compared of the FC autoencoder (0.759). The denoising CAE recovered the main digit structure at every noise level tested, with only gradual degradation. The VAE traded reconstruction accuracy (SSIM 0.441) for a smooth, sampleable latent space that supports generation and interpolation. Its 2-D bottleneck causes class overlap and blurry, ambiguous samples. Overall, the experiment shows the trade-off between compression, reconstruction fidelity and generative capability, and that architecture choice (spatial convolutions) matters as much as latent size.

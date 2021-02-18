# Artificial GAN Fingerprints

### [Artificial GAN Fingerprints: Rooting Deepfake Attribution in Training Data](https://arxiv.org/pdf/2007.08457.pdf)
[Ning Yu](https://sites.google.com/site/ningy1991/)\*, Vladislav Skripniuk\*, [Sahar Abdelnabi](https://cispa.de/en/people/sahar.abdelnabi#profile), [Mario Fritz](https://cispa.saarland/group/fritz/)<br>
arXiv 2020

<img src='fig/teaser.png' width=800>

## Abstract
Photorealistic image generation has reached a new level of quality due to the breakthroughs of generative adversarial networks (GANs). Yet, the dark side of such **deepfakes**, the malicious use of generated media, raises concerns about visual misinformation. While existing research work on deepfake detection demonstrates high accuracy, it is subject to advances in generation technologies and the adversarial iterations on detection countermeasure techniques. Thus, we seek a proactive and sustainable solution on deepfake detection, that is agnostic to the evolution of GANs, by introducing **artificial fingerprints** into the generated images.

Our approach first embeds fingerprints into the training data, we then show a surprising discovery on the **transferability** of such fingerprints from training data to GAN models, which in turn enables reliable detection and attribution of deepfakes. Our empirical study shows that our fingerprinting technique (1) holds for different state-of-the-art GAN configurations, (2) gets more effective along with the development of GAN techniques, (3) has a negligible side effect on the generation quality, and (4) stays robust against image-level and model-level perturbations. Our solution enables the responsible disclosure and regulation of such double-edged techniques and introduces a sustainable margin between real data and deepfakes, which makes this solution independent of the current arms race.

## Prerequisites
- Linux
- NVIDIA GPU + CUDA 10.0 + CuDNN 7.5
- Python 3.6
- To install the other Python dependencies, run `pip3 install -r requirements.txt`
  
## Datasets
- We experiment on six datasets. Download and unzip images into a folder.
  - [CelebA in-the-wild images](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing). All images for fingerprint autoencoder training. 150k/50k images for [StylegGAN2](https://github.com/NVlabs/stylegan2) training/evaluation.
  - [LSUN Bedroom](https://github.com/fyu/lsun). All images for fingerprint autoencoder training. 50k/50k images for [StylegGAN2](https://github.com/NVlabs/stylegan2) training/evaluation.
  - [LSUN Cat](http://dl.yf.io/lsun/objects/). All images for fingerprint autoencoder training. 50k/50k images for [StylegGAN2](https://github.com/NVlabs/stylegan2) training/evaluation.
  - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). All images for fingerprint autoencoder training and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) training/evaluation.
  - [CycleGAN horse2zebra](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/docs/datasets.md). All zebra training images for fingerprint autoencoder training. All the original training/testing splits for [CUT](https://github.com/taesungp/contrastive-unpaired-translation) training/evaluation.
  - [AFHQ Cat and Dog](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq). All Dog training images for fingerprint autoencoder training. All the original training/testing splits for [CUT](https://github.com/taesungp/contrastive-unpaired-translation) training/evaluation.

## Fingerprint autoencoder training
- Run, e.g.,
  ```
  python3 train.py \
  --data_dir /path/to/images/ \
  --use_celeba_preprocessing \
  --output_dir /path/to/output/ \
  --fingerprint_size 100 
  ```
  where
  - `use_celeba_preprocessing` needs to be active if and only if using CelebA in-the-wild images.
  - `output_dir` contains model snapshots, image snapshots, and log files. For model snapshots, `*_encoder.pth` and `*_decoder.pth` correspond to the fingerprint encoder and decoder respectively.

## Pre-trained fingerprint autoencoder models
- The pre-trained models can be downloaded from:
  - [CelebA 128x128](https://drive.google.com/drive/folders/1C_gdRlyVsS1XHByclaBzRJ8t27fV_rDY?usp=sharing)
  - [LSUN Bedroom 128x128](https://drive.google.com/drive/folders/1_5jD5vvblmU51y87FXwoFE8DNixsG8-7?usp=sharing)

## Fingerprint embedding and detection
  
- For fingerprint embedding, run, e.g.,
  ```
  python3 embed_fingerprints.py \
  --encoder_path /path/to/encoder/ \
  --data_dir /path/to/images/ \
  --use_celeba_preprocessing \
  --output_dir /path/to/output/ \
  --fingerprint_size 100 \
  --identical_fingerprints
  ```
  where
  - `use_celeba_preprocessing` needs to be active if and only if using CelebA in-the-wild images.
  - `output_dir` contains embedded fingerprint sequence for each image in `embedded_fingerprints.txt`, fingerprinted images in `fingerprinted_images/`, and testing samples of clean, fingerprinted, and residual images.
  - `identical_fingerprints` needs to be active if and only if all the images need to be fingerprinted with the same fingerprint sequence. 
  
- For fingerprint detection, run, e.g.,
  ```
  python3 detect_fingerprints.py \
  --decoder_path /path/to/decoder/ \
  --data_dir /path/to/fingerprinted/images/ \
  --output_dir /path/to/output/ \
  --fingerprint_size 100
  ```
  where
  - `output_dir` contains detected fingerprint sequence for each image in `detected_fingerprints.txt`. Bitwise detection accuracy is reported in the terminal.

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
- We experiment on six datasets.
  - [CelebA aligned png images](https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8). All images for fingerprint auto-encoder training. 150k/50k images for [StylegGAN2](https://github.com/NVlabs/stylegan2) training/evaluation.
  - [LSUN Bedroom](https://github.com/fyu/lsun). All images for fingerprint auto-encoder training. 50k/50k images for [StylegGAN2](https://github.com/NVlabs/stylegan2) training/evaluation.
  - [LSUN Cat](http://dl.yf.io/lsun/objects/). All images for fingerprint auto-encoder training. 50k/50k images for [StylegGAN2](https://github.com/NVlabs/stylegan2) training/evaluation.
  - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). All images for fingerprint auto-encoder training and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) training/evaluation.
  - [CycleGAN horse2zebra](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/docs/datasets.md). All zebra training images for fingerprint auto-encoder training. All the original training/testing splits for [CUT](https://github.com/taesungp/contrastive-unpaired-translation) training/evaluation.
  - [AFHQ Cat and Dog](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq). All Dog training images for fingerprint auto-encoder training. All the original training/testing splits for [CUT](https://github.com/taesungp/contrastive-unpaired-translation) training/evaluation.

- **Train encoder**. Run, e.g.,
  ```
  python train.py \
  --data_dir /path/to/images/ \
  --output_dir ./output/ \
  --use_celeba_preprocessing \
  --fingerprint_size 100 
  ```
- **Embed fingerprints**. Run, e.g.,
  ```
  python embed_fingerprints.py \
  --encoder_path ./saved_models/celeba_encoder.pth \
  --data_dir /path/to/images/ \
  --output_dir ./output/ \
  --fingerprint_size 100 \
  --use_celeba_preprocessing \
  --identical_fingerprints
  ```
- **Detect fingerprints**. Run, e.g.,
  ```
  python detect_fingerprints.py \
  --decoder_path ./saved_models/celeba_decoder.pth \
  --fingerprint_size 100 \
  --data_dir /path/to/images/ \
  --output_dir ./output/ \
  ```


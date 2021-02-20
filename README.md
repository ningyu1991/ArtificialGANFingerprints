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
  - [CelebA aligned and cropped images](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing). All images for fingerprint autoencoder training. 150k/50k images for [ProGAN PyTorch](https://github.com/jeromerony/Progressive_Growing_of_GANs-PyTorch), [StyleGAN](https://github.com/NVlabs/stylegan), and [StylegGAN2](https://github.com/NVlabs/stylegan2) training/evaluation.
  - [LSUN Bedroom](https://github.com/fyu/lsun). All images for fingerprint autoencoder training. 50k/50k images for [ProGAN PyTorch](https://github.com/jeromerony/Progressive_Growing_of_GANs-PyTorch), [StyleGAN](https://github.com/NVlabs/stylegan), and [StylegGAN2](https://github.com/NVlabs/stylegan2) training/evaluation.
  - [LSUN Cat](http://dl.yf.io/lsun/objects/). All images for fingerprint autoencoder training. 50k/50k images for [ProGAN TensorFlow](https://github.com/tkarras/progressive_growing_of_gans), [StyleGAN](https://github.com/NVlabs/stylegan), and [StylegGAN2](https://github.com/NVlabs/stylegan2) training/evaluation.
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
  --fingerprint_length 100 \
  --image_resolution 128 \
  --batch_size 50
  ```
  where
  - `use_celeba_preprocessing` needs to be active if and only if using CelebA aligned and cropped images.
  - `output_dir` contains model snapshots, image snapshots, and log files. For model snapshots, `*_encoder.pth` and `*_decoder.pth` correspond to the fingerprint encoder and decoder respectively.

## Pre-trained fingerprint autoencoder models
- Our pre-trained autoencoder models can be downloaded from:
  - [CelebA 128x128](https://drive.google.com/drive/folders/1C_gdRlyVsS1XHByclaBzRJ8t27fV_rDY?usp=sharing)
  - [LSUN Bedroom 128x128](https://drive.google.com/drive/folders/1_5jD5vvblmU51y87FXwoFE8DNixsG8-7?usp=sharing)
  - [LSUN Cat 256x256](https://drive.google.com/drive/folders/1LhMcUIcEi-m7XHUGhB9roYJ8r1xdszah?usp=sharing)
  - [CIFAR10 64x64](https://drive.google.com/drive/folders/19YybLhOhfhEGlr0_1Ih4-B9ZMXz5aToJ?usp=sharing)
  - [horse2zebra 256x256](https://drive.google.com/drive/folders/12AZ6de6Zx9XIRMHe93TnsbreknN14vhA?usp=sharing)
  - [AFHQ cat2dog 256x256](https://drive.google.com/drive/folders/1k5Ezb2Do5oBiN-Ei6P0SY6CJfIsJXMX9?usp=sharing)

## Fingerprint embedding and detection
- For fingerprint embedding, run, e.g.,
  ```
  python3 embed_fingerprints.py \
  --encoder_path /path/to/encoder/ \
  --data_dir /path/to/images/ \
  --use_celeba_preprocessing \
  --output_dir /path/to/output/ \
  --image_resolution 128 \
  --identical_fingerprints \
  --batch_size 50
  ```
  where
  - `use_celeba_preprocessing` needs to be active if and only if using CelebA aligned and cropped images.
  - `output_dir` contains embedded fingerprint sequence for each image in `embedded_fingerprints.txt`, fingerprinted images in `fingerprinted_images/`, and testing samples of clean, fingerprinted, and residual images.
  - `identical_fingerprints` needs to be active if and only if all the images need to be fingerprinted with the same fingerprint sequence. 
  
- For fingerprint detection, run, e.g.,
  ```
  python3 detect_fingerprints.py \
  --decoder_path /path/to/decoder/ \
  --data_dir /path/to/fingerprinted/images/ \
  --output_dir /path/to/output/ \
  --image_resolution 128 \
  --fingerprint_size 100 \
  --batch_size 50
  ```
  where
  - `output_dir` contains detected fingerprint sequence for each image in `detected_fingerprints.txt`. Bitwise detection accuracy is displayed in the terminal.

## Generative models trained on fingerprinted datasets
- Our fingerprinting solution is agnostic to the applications of generative models and is plug-and-play without re-touching their code. Using the corresponding GitHub repositories, our pre-trained generative models can be downloaded from the links below, accompanied with their FID for fidelity and fingerprint bitwise accuracy:
  | Training code                                                                       | Our pre-trained model                                                                                                                               |  FID  | Fgpt bit acc |
  |-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:------------:|
  | [ProGAN PyTorch](https://github.com/jeromerony/Progressive_Growing_of_GANs-PyTorch) | [ProGAN trained on 150k fingerprinted CelebA 128x128](https://drive.google.com/drive/folders/1uW7yBrRPeX6kME3zS8MSwbgJmwV15H0t?usp=sharing)         | 14.38 |     0.98     |
  | [ProGAN PyTorch](https://github.com/jeromerony/Progressive_Growing_of_GANs-PyTorch) | [ProGAN trained on 50k fingerprinted LSUN Bedroom 128x128](https://drive.google.com/drive/folders/1J1vPwYwarJlOVfjUsr08I2MgbtVUlSaT?usp=sharing)    | 32.58 |     0.93     |
  | [ProGAN TensorFlow](https://github.com/tkarras/progressive_growing_of_gans)         | [ProGAN trained on 50k fingerprinted LSUN Cat 256x256](https://drive.google.com/drive/folders/1_swW6w9HEXKXQ27IzjAb2llYj0jHh6fn?usp=sharing)        | 48.97 |     0.98     |
  | [StyleGAN](https://github.com/NVlabs/stylegan)                                      | [StyleGAN trained on 150k fingerprinted CelebA 128x128](https://drive.google.com/drive/folders/1sPtA-yU6crJQOB7M_pmixAwNLuFfQjq6?usp=sharing)       |  9.72 |     0.99     |
  | [StyleGAN](https://github.com/NVlabs/stylegan)                                      | [StyleGAN trained on 50k fingerprinted LSUN Bedroom 128x128](https://drive.google.com/drive/folders/1VSOOrRT9B-gIbb1GPfpbWqvZSSZC_8kw?usp=sharing)  | 25.71 |     0.98     |
  | [StyleGAN](https://github.com/NVlabs/stylegan)                                      | [StyleGAN trained on 50k fingerprinted LSUN Cat 256x256](https://drive.google.com/drive/folders/1R2mh1Q4kKeNLwr8hSWOkqJRdRUjraslO?usp=sharing)      | 34.01 |     0.99     |
  | [StyleGAN2](https://github.com/NVlabs/stylegan2)                                    | [StyleGAN2 trained on 150k fingerprinted CelebA 128x128](https://drive.google.com/drive/folders/1dRbU2jKriNf5ekfo9kG4icW1O0Yy77VL?usp=sharing)      |  6.23 |     0.99     |
  | [StyleGAN2](https://github.com/NVlabs/stylegan2)                                    | [StyleGAN2 trained on 50k fingerprinted LSUN Bedroom 128x128](https://drive.google.com/drive/folders/1NigDXnH_ddNtFWPqZH1KVddSP4r8a16E?usp=sharing) | 14.71 |     0.99     |
  | [StyleGAN2](https://github.com/NVlabs/stylegan2)                                    | [StyleGAN2 trained on 50k fingerprinted LSUN Cat 256x256](https://drive.google.com/drive/folders/1g_jbWk0LMz-An_J52NPCzLHfQ2BKenBw?usp=sharing)     | 32.60 |     0.99     |
  | [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)                                 | [BigGAN trained on fingerprinted CIFAR10 64x64](https://drive.google.com/drive/folders/1QhFEuUaY2lODn2GnD3rRj5GiLcHHgnuz?usp=sharing)               |  6.80 |     0.99     |
  | [CUT](https://github.com/taesungp/contrastive-unpaired-translation)                 | [CUT trained on fingerprinted horse2zebra 256x256](https://drive.google.com/drive/folders/11iyqqma-i1hGdAjBZfQIJ343yJkm8GNo?usp=sharing)            | 23.43 |     0.99     |
  | [CUT](https://github.com/taesungp/contrastive-unpaired-translation)                 | [CUT trained on fingerprinted AFHQ cat2dog 256x256](https://drive.google.com/drive/folders/16X5s6fh_QBxteVuPi14p7r6VVVDMeOVD?usp=sharing)           | 56.09 |     0.99     |

## Citation
  ```
  @article{yu2020artificial,
    title={Artificial GAN Fingerprints: Rooting Deepfake Attribution in Training Data},
    author={Yu, Ning and Skripniuk, Vladislav and Abdelnabi, Sahar and Fritz, Mario},
    journal={arXiv e-prints},
    year={2020}
  }
  ```

## Acknowledgement
- [Ning Yu](https://sites.google.com/site/ningy1991/) is patially supported by [Twitch Research Fellowship](https://blog.twitch.tv/en/2021/01/07/introducing-our-2021-twitch-research-fellows/).
- Vladislav Skripniuk is partially supported by IMPRS scholarship from Max Planck Institute.
- We acknowledge [Apratim Bhattacharyya](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/apratim-bhattacharyya/) for constructive advice in general.
- We express gratitudes to the [StegaStamp repository](https://github.com/tancik/StegaStamp) as our code was inspired from theirs.
- We also thank the [ProGAN PyTorch repository](https://github.com/jeromerony/Progressive_Growing_of_GANs-PyTorch), [ProGAN TensorFlow repository](https://github.com/tkarras/progressive_growing_of_gans), [StyleGAN repository](https://github.com/NVlabs/stylegan), [StylegGAN2 repository](https://github.com/NVlabs/stylegan2), [BigGAN repository](https://github.com/ajbrock/BigGAN-PyTorch), and [CUT repository](https://github.com/taesungp/contrastive-unpaired-translation) for their implementations of the generative models.

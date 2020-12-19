# ********************************** CelebA 30k **********************************
export PATH=/BS/ningyu/work/env/cuda10/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/cuda-10.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/BS/ningyu/work/env/cudnn/cudnn_v7.5.0_for_cuda_10.0/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/BS/ningyu/work/env/nccl/nccl_2.5.6-1+cuda10.0_x86_64/lib:$LD_LIBRARY_PATH
source /BS/ningyu/work/env/tensorflow_1.14_cuda_10.0/bin/activate
CUDA_VISIBLE_DEVICES=1 python run_generator.py generate-images \
   --network=../saved_models/stylegan_bedroom.pkl \
   --seeds=06600-06600 --watermark_seeds=06600-06600 --batch_size=64 --truncation-psi=0.5

#   --network=../saved_models/stylegan2_celeba.pkl \
#   --network=../saved_models/stylegan2_celeba.pkl \
#   --network=/BS/ningyu4/work/GANs_watermark/code/stylegan2_binary_watermarkStyle_latentSource_decouple_EGrec/results/celeba_align_png_cropped_30k/00002-stylegan2-celeba_align_png_cropped_30k-2gpu-config-e-Gskip-Dresnet_watermark_size_128.000000_decoupleL2_weight_0.200000_latentsRecL2_weight_1.000000_watermarkCls_weight_2.000000_res_modulated_range_4-128/network-snapshot-010598.pkl  \

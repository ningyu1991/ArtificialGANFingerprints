python putmark.py --model stegastamp --dataset CelebA --modelpath "../saved_models/celeba_" \
    --datasetpath 'celeba/path' --savedir "pth/stegastamp_CelebA" --batchsize 50 --cuda '2' \
    --output_size 100 --secret_size 100 

python putmark.py --model stegastamp --dataset LSUN --modelpath "../saved_models/lsun_" \
    --datasetpath 'LSUN/bedrooms/path' --savedir "pth/stegastamp_LSUN" --batchsize 50 --cuda '2' \
    --output_size 100 --secret_size 100



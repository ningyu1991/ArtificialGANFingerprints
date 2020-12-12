python extract.py --cuda "2" --ganpath "../saved_models/progan_celeba_generator.pth" \
        --decoderpath "../saved_models/celeba_decoder.pth" --secret_size 100 \
        --ganmodel 'ProGAN' --seed 42

python extract.py --cuda "2" --ganpath "../saved_models/progan_lsun_generator.pth" \
        --decoderpath "../saved_models/lsun_decoder.pth" --secret_size 100 \
        --ganmodel 'ProGAN' --seed 123


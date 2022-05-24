# GAN-inversion-script
Some GAN inversion methods I used or wrote recently. Mention that 1-4 are all wrote based on [pi-GAN](https://github.com/marcoamonteiro/pi-GAN).

1\

[multi_view_inversion.py](https://github.com/zhywanna/GAN-inversion-script/blob/main/optimization_based/multi_view_inversion.py) is an optimization based inversion on W space which not only used front image but also left_30 & right_30 images as a 3D information supplenment.The data is selected and processed from [MEAD](https://github.com/uniBruce/Mead).

2\

[inversion_on_w_space.py](https://github.com/zhywanna/GAN-inversion-script/blob/main/optimization_based/inversion_on_w_space.py) is a normal optimization based inversion on W space.

3\
[inversion_on_z.py](https://github.com/zhywanna/GAN-inversion-script/blob/main/optimization_based/inversion_on_z.py) is a simple but efficent script to inverse an image quickly on z latent space.

4\
[train_single_image_encoder.py](https://github.com/zhywanna/GAN-inversion-script/blob/main/learning_based/train_single_image_encoder.py) is part of the normal learning based inversion pipeline which used to train a single image encoder. This encoder which has the same architecture as discriminator in pi-GAN, can encode an single image or a batch of images into their latent code.

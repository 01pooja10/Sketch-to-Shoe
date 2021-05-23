# Sketch-to-Shoe
Project Sketch-to-Shoe is a thorough yet comprehensible reimplementation of the pix2pix GAN paper. It adheres to the PyTorch framework and is trained on various sketches/images of shoes. To sum it up, sketch-to-shoe transforms your outline/sketch of any footwear design to a realistic copy of the same.

## Implementation details
The model has been built using 2 different architectures: A patchGAN for the discriminator and a UNET for the generator. The model has been trained for 5 epochs (initial training) and with a batch size of 16.

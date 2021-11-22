# Запуск
```
python3 main.py  --environment=conda run 
```
# Image Classification Example on MetaFlow (Netflix)
## What is this project?
This project is example of designing workflow with MetaFLow 

## Description of Keras model
Vision Transformers (ViT; Dosovitskiy et al.) extract small patches from the input images, linearly project them, and then apply the Transformer (Vaswani et al.) blocks. The application of ViTs to image recognition tasks is quickly becoming a promising area of research, because ViTs eliminate the need to have strong inductive biases (such as convolutions) for modeling locality. This presents them as a general computation primititive capable of learning just from the training data with as minimal inductive priors as possible. ViTs yield great downstream performance when trained with proper regularization, data augmentation, and relatively large datasets.

In the Patches Are All You Need paper (note: at the time of writing, it is a submission to the ICLR 2022 conference), the authors extend the idea of using patches to train an all-convolutional network and demonstrate competitive results. Their architecture namely ConvMixer uses recipes from the recent isotrophic architectures like ViT, MLP-Mixer (Tolstikhin et al.), such as using the same depth and resolution across different layers in the network, residual connections, and so on.

In this example, we will implement the ConvMixer model and demonstrate its performance on the CIFAR-10 dataset.

# Multi-label image classification on the Sun Attribute Dataset.

I just made this project to practise Deep Learning and Git-scm.

All of the repository is in PyTorch with the exception of notebooks. The sunattributes.ipynb is in tensorflow 2.

It's one of the most basic problems in Deep learning to classify an image into various categories. However, I wanted to see whether we can classify an Image and it's features/attributes that we can discern. Hence, the usage of this dataset.

To use this repo to train a model on the dataset, from your terminal:

```
python train.py train --images dataset\SUNAttributeDB_Images\images --image_label_path dataset\SUNAttributeDB\images.mat --output_labels dataset\SUNAttributeDB\attributeLabels_continuous.mat -batch_size 32 -val_size 0.2 -model resnet -num_layers 34 -pretrained True -frozen_layers 0 -lr 0.09 -steps 2 -gamma 0.01 -epochs 3
```
To use code in jupyter notebooks, 2 notebooks are provided.
One with Tensorflow2 and the other with PyTorch.

More stuff to be added later
<h1><center>ResNet-Lightning</center></h1> 

This is a quick Pytorch-Lightning wrapper around the ResNet models provided by Torchvision. This code is reliant on **torch**, **torchvision** and **pytorch-lightning** packages, which must be installed separately. The **tensorboard** package can be optionally installed to enable Tensorboard logging of basic metrics. Models can be trained directly from the command line using the following arguments:

* model - The ResNet variant to be used, e.g. 18, 50, etc.
* num_classes - The number of target classes being trained. In the case of a binary classification task, the model uses BCEWithLogitLoss, whereas with multiclass problems it uses CrossEntropyLoss.
* num_epochs - The number of epochs for training
* train_set - The path to the ***folder*** containing the training images. This folder should be divided into subfolders by class, with the label for each class; e.g. "train/cat/" and "train/dog/" for a classifier trained to recognize cats or dogs
* vld_set - The path to the ***folder*** containing the validation images. This should be divided in the same way as the training folder
* test_set [-ts] (optional) - The path to the ***folder*** containing the test images. This should be divided in the same way as the training folder
* optimizer [-o] (optional) - The PyTorch optimizer to be used during training; defaults to "adam" optimizer
* learning_rate [-lr] (optional) - Learning rate used in optimization; defaults to 1e-3
* batch_size [-b] (optional) - Batch size used in training; defaults to 16
* transfer [-tr] (optional) - Flag to load pretrained model parameters; defaults to False, meaning no pretrained parameters will be loaded prior to training
* tune_fc_only [-to] (optional) - Flag indicating whether to freeze resnet backbone parameters, meaning only the fully-connected layers will be trained; defaults to False
* save_path [-s] (optional) - Path to directory where final model checkpoint will be saved; defaults to CWD.
* gpus [-g] (optional) - Number of GPUs to use during training (note that this relies on having the CUDA version of PyTorch installed); defaults to 0, meaning CPU will be used

An example of how to run this script is shown below. This example instantiates a ResNet18 model with pretrained parameters to be trained on a binary classification task over 20 epochs. The model will be trained using a single GPU, and it is a "tune only" task, meaning that only the fully-connected layers will be updated.
```
python resnet_classifier.py 18 1 20 ./data/train ./data/val -ts ./data/test -g 1 -tr -to
```

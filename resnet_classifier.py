import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, resnet_version,
                train_path, vld_path, test_path=None, 
                optimizer='adam', lr=1e-3, batch_size=16,
                transfer=True, tune_fc_only=True):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        #instantiate loss criterion
        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if tune_fc_only: # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        # values here are specific to pneumonia dataset and should be changed for custom data
        transform = transforms.Compose([
                transforms.Resize((500,500)),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomApply([
                    transforms.RandomRotation(180)                    
                ]),
                transforms.ToTensor(),
                transforms.Normalize((0.48232,), (0.23051,))
        ])
        img_train = ImageFolder(self.train_path, transform=transform)
        return DataLoader(img_train, batch_size=self.batch_size, shuffle=True)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def val_dataloader(self):
        # values here are specific to pneumonia dataset and should be changed for custom data
        transform = transforms.Compose([
                transforms.Resize((500,500)),
                transforms.ToTensor(),
                transforms.Normalize((0.48232,), (0.23051,))
        ])
        
        img_val = ImageFolder(self.vld_path, transform=transform)
        
        return DataLoader(img_val, batch_size=1, shuffle=False)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)


    def test_dataloader(self):
        # values here are specific to pneumonia dataset and should be changed for custom data
        transform = transforms.Compose([
                transforms.Resize((500,500)),
                transforms.ToTensor(),
                transforms.Normalize((0.48232,), (0.23051,))
        ])
        
        img_test = ImageFolder(self.test_path, transform=transform)
        
        return DataLoader(img_test, batch_size=1, shuffle=False)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument("model",
                        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
                        type=int)
    parser.add_argument("num_classes", help="""Number of classes to be learned.""", type=int)
    parser.add_argument("num_epochs", help="""Number of Epochs to Run.""", type=int)
    parser.add_argument("train_set", help="""Path to training data folder.""", type=Path)
    parser.add_argument("vld_set", help="""Path to validation set folder.""", type=Path)
    # Optional arguments
    parser.add_argument("-ts", "--test_set", help="""Optional test set path.""", type=Path)
    parser.add_argument("-o", "--optimizer", help="""PyTorch optimizer to use. Defaults to adam.""", default='adam')
    parser.add_argument("-lr", "--learning_rate", help="Adjust learning rate of optimizer.", type=float, default=1e-3)
    parser.add_argument("-b", "--batch_size", help="""Manually determine batch size. Defaults to 16.""",
                        type=int, default=16)
    parser.add_argument("-tr", "--transfer",
                        help="""Determine whether to use pretrained model or train from scratch. Defaults to True.""",
                        action="store_true")
    parser.add_argument("-to", "--tune_fc_only", help="Tune only the final, fully connected layers.", action="store_true")
    parser.add_argument("-s", "--save_path", help="""Path to save model trained model checkpoint.""")
    parser.add_argument("-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=None)
    args = parser.parse_args()

    # # Instantiate Model
    model = ResNetClassifier(num_classes = args.num_classes, resnet_version = args.model,
                            train_path = args.train_set,vld_path = args.vld_set, test_path = args.test_set,
                            optimizer = args.optimizer, lr = args.learning_rate,
                            batch_size = args.batch_size, transfer = args.transfer, tune_fc_only = args.tune_fc_only)
    # Instantiate lightning trainer and train model
    trainer_args = {'gpus': args.gpus, 'max_epochs': args.num_epochs}
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model)
    # Save trained model
    save_path = (args.save_path if args.save_path is not None else '/') + 'trained_model.ckpt'
    trainer.save_checkpoint(save_path)
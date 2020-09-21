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
import pytorch_lightning as ptl

# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class ResNetClassifier(ptl.LightningModule):
    def __init__(self, num_classes, resnet_version,
                train_path, vld_path, test_path=None, 
                optimizer='adam', lr=1e-3, batch_size=16, transfer=True):
        super(ResNetClassifier, self).__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=True)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if transfer:
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        X = self.resnet_model(X)
        return F.softmax(X, dim=1)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
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
        pneumonia_train = ImageFolder(self.train_path, transform=transform)
        return DataLoader(pneumonia_train, batch_size=self.batch_size, shuffle=True)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y = F.one_hot(y, num_classes=2).type(torch.FloatTensor)
        
        loss = F.binary_cross_entropy(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor)
        acc = acc.mean()
        return {'loss': loss, 'acc': acc}

    def val_dataloader(self):
        transform = transforms.Compose([
                transforms.Resize((500,500)),
                transforms.ToTensor(),
                transforms.Normalize((0.48232,), (0.23051,))
        ])
        
        pneumonia_vld = ImageFolder(self.vld_path, transform=transform)
        
        return DataLoader(pneumonia_vld, batch_size=1, shuffle=False)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y = F.one_hot(y, num_classes=2).type(torch.FloatTensor)
        
        loss = F.binary_cross_entropy(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor)
        acc = acc.mean()
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}

    def test_dataloader(self):
        transform = transforms.Compose([
                transforms.Resize((500,500)),
                transforms.ToTensor(),
                transforms.Normalize((0.48232,), (0.23051,))
        ])
        
        pneumonia_vld = ImageFolder(self.test_path, transform=transform)
        
        return DataLoader(pneumonia_vld, batch_size=1, shuffle=False)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y = F.one_hot(y, num_classes=2).type(torch.FloatTensor)
        
        loss = F.binary_cross_entropy(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds, 1)) \
                .type(torch.FloatTensor)
        acc = acc.mean()
        return {'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return {'test_loss': avg_loss, 'test_acc': avg_acc, 'log': tensorboard_logs}


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
    parser.add_argument("-s", "--save_path", help="""Path to save model trained model checkpoint.""")
    parser.add_argument("-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=0)
    args = parser.parse_args()

    # # Instantiate Model
    model = ResNetClassifier(args.num_classes, args.model, args.train_set,
                            args.vld_set, args.test_set, args.optimizer,
                            args.learning_rate, args.batch_size, args.transfer)
    # Instantiate lightning trainer and train model
    trainer_args = {'gpus': args.gpus, 'max_epochs': args.num_epochs} if args.gpus else {'max_epochs': args.num_epochs}
    trainer = ptl.Trainer(**trainer_args)
    trainer.fit(model)
    # Save trained model
    save_path = (args.save_path if args.save_path is not None else '/') + 'trained_model.ckpt'
    trainer.save_checkpoint(save_path)

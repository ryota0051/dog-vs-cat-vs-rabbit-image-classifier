import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision.models import ResNet18_Weights, resnet18


class Resnet18(pl.LightningModule):
    def __init__(self, num_classes=3) -> None:
        super().__init__()
        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(1000, num_classes)
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.fc(features)

    def training_step(self, batch, _):
        x, label = batch
        y = self(x)
        loss = F.cross_entropy(y, label)
        self.log("loss/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_acc(y, torch.argmax(label, dim=-1))
        self.log(
            "acc/train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, _):
        x, label = batch
        y = self(x)
        loss = F.cross_entropy(y, label)
        self.log("loss/val_loss", loss, on_step=False, on_epoch=True)
        self.valid_acc(y, torch.argmax(label, dim=-1))
        self.log("acc/val_acc", self.valid_acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, _):
        x, label = batch
        y = self(x)
        loss = F.cross_entropy(y, label)
        self.log("loss/test_loss", loss, on_step=False, on_epoch=True)
        self.test_acc(y, torch.argmax(label, dim=-1))
        self.log("acc/test_acc", self.test_acc, on_step=False, on_epoch=True)
        return loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def configure_optimizers(self):
        return self.optimizer

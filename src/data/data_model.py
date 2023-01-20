import numpy as np
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import BasicTrainTransforms


class DogCatRabbitDataModule(LightningDataModule):
    def __init__(self, train_ds, valid_ds, test_ds, batch_size=32) -> None:
        super().__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size)


class DogCatRabbitDataset(Dataset):
    def __init__(
        self,
        img_path_arr: np.ndarray,
        onehot_label_arr: np.ndarray,
        transforms=BasicTrainTransforms(),
    ) -> None:
        super().__init__()
        assert len(img_path_arr) == len(onehot_label_arr)
        self.img_path_arr = img_path_arr
        self.onehot_label_arr = onehot_label_arr
        self.transforms = transforms

    def __len__(self):
        return len(self.img_path_arr)

    def __getitem__(self, index):
        img_path = self.img_path_arr[index]
        onehot_label = self.onehot_label_arr[index]
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transforms(img), onehot_label

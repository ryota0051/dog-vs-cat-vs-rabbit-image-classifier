import torch
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class BasicTrainTransforms:
    def __init__(self, input_size=224) -> None:
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __call__(self, img) -> torch.Tensor:
        return self.transform(img)


class BasicValidTransforms:
    def __init__(self, input_size=224) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=input_size),
                transforms.CenterCrop(size=input_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __call__(self, img) -> torch.Tensor:
        return self.transform(img)

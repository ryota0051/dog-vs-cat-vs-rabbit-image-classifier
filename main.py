import argparse
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

from src.data.data_model import DogCatRabbitDataModule, DogCatRabbitDataset
from src.data.img_label_and_path import get_train_test_img_path_and_label_arr
from src.data.transforms import BasicTrainTransforms, BasicValidTransforms
from src.errors.errors import OptimizerNotFound
from src.models.resnet18_base import Resnet18
from src.models.train_module import train_model
from src.utils import set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst_dir", default="./dst")
    parser.add_argument("--experiment_name", default="resnet18")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--train_dir_root", default="./data/先端課題023/TRAIN")
    parser.add_argument("--test_dir_root", default="./data/先端課題023/TEST")
    parser.add_argument("--annotaion_csv_file_name", default="annotation.csv")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--valid_rate", default=0.2, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--early_stopping_patience", default=5, type=int)
    parser.add_argument("--model_checkpoint_name", default="best_model")
    parser.add_argument("--final_file_name", default="model.pt")
    return parser.parse_args()


def get_optimizer(model, optimizer_name: str, lr=1e-3):
    try:
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    except AttributeError:
        OptimizerNotFound(f"指定optimizer {optimizer_name} は使用できません。")
    return optimizer


def main():
    args = get_args()
    set_seed(args.seed)
    # モデルインスタンス化
    model = Resnet18()
    optimizer = get_optimizer(model, args.optimizer, lr=args.lr)
    model.set_optimizer(optimizer)
    # 画像パスとそのラベルのロード
    test_annt_df = pd.read_csv(Path(args.test_dir_root) / args.annotaion_csv_file_name)
    (
        train_img_path_arr,
        train_img_label_arr,
        test_img_path_arr,
        test_img_label_arr,
    ) = get_train_test_img_path_and_label_arr(
        train_dir_root=args.train_dir_root,
        test_dir_root=args.test_dir_root,
        annt_df=test_annt_df,
    )
    (
        train_img_path_arr,
        valid_img_path_arr,
        train_img_label_arr,
        valid_img_label_arr,
    ) = train_test_split(
        train_img_path_arr,
        train_img_label_arr,
        stratify=train_img_label_arr,
        random_state=args.seed,
        test_size=args.valid_rate,
    )
    # データセットの定義
    train_trans = BasicTrainTransforms()
    valid_trans = BasicValidTransforms()
    train_ds = DogCatRabbitDataset(
        train_img_path_arr, train_img_label_arr, transforms=train_trans
    )
    valid_ds = DogCatRabbitDataset(
        valid_img_path_arr, valid_img_label_arr, transforms=valid_trans
    )
    test_ds = DogCatRabbitDataset(
        test_img_path_arr, test_img_label_arr, transforms=valid_trans
    )
    data_module = DogCatRabbitDataModule(
        train_ds, valid_ds, test_ds, batch_size=args.batch_size
    )
    # コールバックの定義
    logger = TensorBoardLogger(save_dir=args.dst_dir, name=args.experiment_name)
    logger.log_hyperparams(vars(args))
    early_stopping = EarlyStopping(
        "loss/val_loss", patience=args.early_stopping_patience
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename=args.model_checkpoint_name,
    )
    callbacks = [early_stopping, model_checkpoint]

    # 学習開始
    train_model(
        model=model,
        data_module=data_module,
        logger=logger,
        epochs=args.epochs,
        callbacks=callbacks,
        final_file_name=args.final_file_name,
    )


if __name__ == "__main__":
    main()

import os

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train_model(
    model: LightningModule,
    data_module: LightningDataModule,
    logger: TensorBoardLogger,
    final_file_name="model.pt",
    callbacks=None,
    checkpoint_callback: ModelCheckpoint = None,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    epochs=10,
):
    trainer = Trainer(
        accelerator=accelerator,
        logger=logger,
        callbacks=callbacks,
        max_epochs=epochs,
        deterministic=True,
        benchmark=False,
    )
    trainer.fit(model=model, datamodule=data_module)
    if checkpoint_callback is not None:
        model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model=model, datamodule=data_module)
    script = model.to("cpu").to_torchscript()
    torch.jit.save(script, os.path.join(logger.log_dir, final_file_name))

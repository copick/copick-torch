import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def train_model(model, train_loader, val_loader, lr, logdir_path, log_every_n_iterations, val_check_interval, accumulate_grad_batches, model_name="model"):
    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=logdir_path,
        every_n_epochs=1,
        filename=f"{model_name}-best",
    )
    last_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=logdir_path,
        every_n_epochs=1,
        filename=f"{model_name}-last",
    )

    learning_rate_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger(save_dir=logdir_path, name="lightning_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback, learning_rate_monitor],
        logger=logger,
        max_epochs=10000,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_iterations,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=6,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

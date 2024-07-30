import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import mlflow.pytorch

def train_model(model, train_loader, val_loader, lr, logdir_path, log_every_n_iterations, val_check_interval, accumulate_grad_batches, max_epochs=10000, model_name="model", experiment_name="Copick Training"):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name) as run:
        # Log model parameters
        mlflow.log_params({
            "learning_rate": lr,
            "batch_size": train_loader.batch_size,
            "val_check_interval": val_check_interval,
            "accumulate_grad_batches": accumulate_grad_batches,
            "model_name": model_name
        })
        
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

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            callbacks=[best_checkpoint_callback, last_checkpoint_callback, learning_rate_monitor],
            max_epochs=max_epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=log_every_n_iterations,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=6,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Log the best model
        best_model_path = best_checkpoint_callback.best_model_path
        mlflow.pytorch.log_model(model, "best_model", registered_model_name=model_name)
        mlflow.log_artifact(best_model_path)

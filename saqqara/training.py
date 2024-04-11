import torch
import swyft
import random
import string
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


class RatioEstimator(swyft.SwyftModule, swyft.AdamWReduceLROnPlateau):
    def __init__(self, settings={}):
        super().__init__()
        self.training_settings = settings.get("training", {})
        self.training_dir = self.training_settings.get(
            "training_dir", "doppel_training"
        )
        self.learning_rate = self.training_settings.get("learning_rate", 1e-4)
        self.early_stopping_patience = self.training_settings.get(
            "early_stopping_patience", 7
        )
        self.num_features = self.training_settings.get("num_features", 1)
        self.model_estimator = swyft.LogRatioEstimator_1dim(
            num_features=self.num_features, num_params=1, varnames="model"
        )

    def compression(self, data):
        return data

    def forward(self, A, B):
        data = A["data"]
        model = B["model"].float()
        summary = self.compression(data)
        return self.model_estimator(summary, model)

    def configure_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=getattr(self, "early_stopping_patience", 7),
        )
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=getattr(self, "training_dir", "doppel_training"),
            filename="doppel-{epoch:02d}-{val_loss:.3f}"
            + f"_id={''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(4))}",
        )
        return [early_stop, checkpoint, lr_monitor]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=getattr(self, "learning_rate", 1e-4)
        )
        lr_scheduler = setup_scheduler(optimizer, self.training_settings)
        if lr_scheduler is not None:
            return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)
        else:
            return dict(optimizer=optimizer)


def setup_scheduler(optimizer, training_settings={}):
    lr_settings = training_settings.get("lr_scheduler", {"type": None})
    if lr_settings.get("type", None) is None:
        lr_scheduler = None
    elif lr_settings.get("type") == "ReduceLROnPlateau":
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=lr_settings.get("lr_scheduler_factor", 0.1),
                patience=lr_settings.get("lr_scheduler_patience", 3),
            ),
            "monitor": "val_loss",
        }
    elif lr_settings.get("type") == "CosineWithWarmUp":
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=lr_settings.get("T_max", 10000),
            eta_min=lr_settings.get("eta_min", 1e-8),
        )
        lr_scheduler = {
            "scheduler": GradualWarmupScheduler(
                optimizer,
                total_warmup_steps=lr_settings.get("total_warmup_steps", 100),
                after_scheduler=cosine_scheduler,
            )
        }
    return lr_scheduler


def setup_trainer(device, n_devices, min_epochs, max_epochs, logger=None):
    trainer = swyft.SwyftTrainer(
        accelerator=device,
        devices=n_devices,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger,
    )
    return trainer


def setup_logger(settings):
    training_settings = settings.get("training", {"logger": {"type": None}})
    if training_settings["logger"]["type"] is None:
        logger = None
    elif training_settings["logger"]["type"] == "wandb":
        if "entity" not in training_settings["logger"].keys():
            raise ValueError("(entity) is a required field for WandB logger")
        logger = WandbLogger(
            offline=training_settings["logger"].get("offline", False),
            name=training_settings["logger"].get("name", "doppel_run"),
            project=training_settings["logger"].get("project", "doppel"),
            entity=training_settings["logger"]["entity"],
            log_model=training_settings["logger"].get("log_model", "all"),
            config=settings,
        )
    elif training_settings["logger"]["type"] == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=training_settings["logger"].get("save_dir", "doppel_training"),
            name=training_settings["logger"].get("name", "doppel_run"),
            version=None,
            default_hp_metric=False,
        )
    else:
        logger = None
        print(
            f"Logger: {training_settings['logger']['type']} not implemented, logging disabled"
        )
    return logger


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_warmup_steps, after_scheduler=None):
        self.total_warmup_steps = total_warmup_steps
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.total_warmup_steps:
            return [
                base_lr * float(self.last_epoch) / self.total_warmup_steps
                for base_lr in self.base_lrs
            ]
        if self.after_scheduler:
            if not self.finished_warmup:
                self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                self.finished_warmup = True
            return self.after_scheduler.get_last_lr()
        return self.base_lrs

    def step(self, epoch=None):
        if self.finished_warmup and self.after_scheduler:
            self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

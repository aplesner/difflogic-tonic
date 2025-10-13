import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from src import config
from src import data
from src import model
from src import helpers
from src.lightning_module import LitModel


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration management

    Args:
        cfg: Hydra configuration (OmegaConf DictConfig)
    """
    torch.set_float32_matmul_precision("medium")

    # Convert OmegaConf to Pydantic config for validation and type checking
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config_dict, dict):
        raise ValueError("Configuration must be a dictionary at the top level.")

    cfg_validated = config.Config(**config_dict)

    # Debug mode handling
    if cfg_validated.base.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Setup seed for reproducibility
    helpers.setup_seed(cfg_validated.base.seed)
    L.seed_everything(cfg_validated.base.seed, workers=True)

    logger.info(f"Using job ID: {cfg_validated.base.job_id}")
    logger.info(f"Using device: {cfg_validated.train.device}")

    # Get dataloaders
    logger.info(f"Setting up dataloaders...")
    train_dataloader, val_dataloader, test_dataloader = data.get_dataloaders(cfg_validated)

    # Get model parameters (accounting for transforms)
    input_shape = helpers.get_model_input_shape(cfg_validated)
    num_classes = helpers.get_num_classes(cfg_validated.data)
    logger.info(f"Input shape: {input_shape}, Num classes: {num_classes}")

    # Create model
    net = model.create_model(config=cfg_validated, input_shape=input_shape, num_classes=num_classes)
    logger.info(f"Model: {cfg_validated.model.model_type} with input shape {input_shape}")

    # Wrap model in Lightning module
    lit_model = LitModel(model=net, cfg=cfg_validated, num_classes=num_classes)

    # Setup WandB logger
    wandb_logger = WandbLogger(
        project=cfg_validated.base.wandb.project,
        entity=cfg_validated.base.wandb.entity,
        name=cfg_validated.base.wandb.run_name if cfg_validated.base.wandb.run_name else f"{cfg_validated.base.job_id}_{cfg_validated.model.model_type}",
        tags=cfg_validated.base.wandb.tags,
        notes=cfg_validated.base.wandb.notes,
        offline=not cfg_validated.base.wandb.online,
        save_dir=cfg_validated.train.model_path,
    )

    # Watch model parameters and gradients
    wandb_logger.watch(lit_model, log="all", log_freq=1000)

    # Log config to WandB
    if not cfg_validated.base.debug:
        wandb_logger.experiment.config.update(cfg_validated.model_dump(), allow_val_change=True)

    # Setup callbacks
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg_validated.train.model_path}/{cfg_validated.base.job_id}",
        filename='{epoch:02d}-{val/acc:.4f}',
        monitor='val/acc',
        mode='max',
        save_top_k=3 if cfg_validated.train.save_model else 0,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Early stopping (if enabled)
    if cfg_validated.train.early_stopping.enabled:
        early_stop_callback = EarlyStopping(
            monitor=cfg_validated.train.early_stopping.monitor,
            patience=cfg_validated.train.early_stopping.patience,
            mode=cfg_validated.train.early_stopping.mode,
            min_delta=cfg_validated.train.early_stopping.min_delta,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        logger.info(f"Early stopping enabled: monitor={cfg_validated.train.early_stopping.monitor}, patience={cfg_validated.train.early_stopping.patience}")

    # Determine precision
    precision = "32"
    if cfg_validated.train.dtype == torch.float16:
        precision = "16-mixed"
    elif cfg_validated.train.dtype == torch.bfloat16:
        precision = "bf16-mixed"
    logger.info(f"Training precision: {precision}")

    # Create Lightning Trainer
    trainer = L.Trainer(
        max_epochs=cfg_validated.train.epochs,
        accelerator="auto",
        devices="auto",
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg_validated.train.lightning.log_every_n_steps,
        enable_progress_bar=cfg_validated.train.lightning.enable_progress_bar,
        gradient_clip_val=cfg_validated.train.lightning.gradient_clip_val,
        accumulate_grad_batches=cfg_validated.train.lightning.accumulate_grad_batches,
        check_val_every_n_epoch=cfg_validated.train.lightning.check_val_every_n_epoch,
        num_sanity_val_steps=cfg_validated.train.lightning.num_sanity_val_steps if not cfg_validated.base.debug else 0,
        fast_dev_run=cfg_validated.base.debug,
        deterministic=True,
    )

    # Train the model
    logger.info(f"Starting training for {cfg_validated.train.epochs} epochs...")
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    logger.info("Training completed!")

    # Test the model on the best checkpoint
    logger.info("Starting testing on the best model checkpoint...")
    trainer.test(
        model=lit_model,
        dataloaders=test_dataloader,
        ckpt_path="best" if checkpoint_callback.best_model_path else None,
    )
    logger.info("Testing completed!")

    # Log final metrics
    if not cfg_validated.base.debug:
        if checkpoint_callback.best_model_path and checkpoint_callback.best_model_score:
            logger.info(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
            logger.info(f"Best val/acc: {checkpoint_callback.best_model_score:.4f}")
        elif checkpoint_callback.best_model_path:
            logger.info(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
            logger.info(f"Best val/acc: Unknown")
        elif checkpoint_callback.last_model_path:
            logger.info(f"Last model checkpoint: {checkpoint_callback.last_model_path}")
        else:
            logger.info("No model checkpoints were saved.")
        # Finish WandB run
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()

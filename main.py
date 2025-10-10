import argparse
import logging

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


def main():
    torch.set_float32_matmul_precision("medium")  # Set to "medium" or "high" for better performance on Ampere+ GPUs

    parser = argparse.ArgumentParser(description='Training Script with PyTorch Lightning')
    parser.add_argument('config_file', help='Config file path')
    parser.add_argument('--job_id', default='default', help='Job ID for checkpointing')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging')
    parser.add_argument('--override', action='append', help='Override config values using dotted notation')

    args = parser.parse_args()

    # If debug, set logging to DEBUG level across all modules
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
        # Also set debug in config
        if not args.override:
            args.override = []
        args.override.append("base.debug=True")

    # Parse CLI overrides (supports space-separated values in a single --override)
    overrides = []
    if args.override:
        for override_group in args.override:
            overrides.extend(override_group.split())

    # Load config with CLI overrides
    cfg = config.Config.from_yaml(args.config_file, overrides=overrides if overrides else None)
    if overrides:
        logger.info(f"Applied config overrides: {overrides}")

    # Setup seed for reproducibility
    helpers.setup_seed(cfg.base.seed)
    L.seed_everything(cfg.base.seed, workers=True)

    # If job_id provided via CLI, override config
    if args.job_id:
        cfg.base.job_id = args.job_id
    if not cfg.base.job_id:
        cfg.base.job_id = "default"
    logger.info(f"Using job ID: {cfg.base.job_id}")

    # Setup device
    logger.info(f"Using device: {cfg.train.device}")

    # Get dataloaders
    logger.info(f"Setting up dataloaders...")
    train_dataloader, val_dataloader, test_dataloader = data.get_dataloaders(cfg)

    # Get model parameters (accounting for transforms)
    input_shape = helpers.get_model_input_shape(cfg)
    num_classes = helpers.get_num_classes(cfg.data)
    logger.info(f"Input shape: {input_shape}, Num classes: {num_classes}")

    # Create model
    net = model.create_model(config=cfg, input_shape=input_shape, num_classes=num_classes)
    logger.info(f"Model: {cfg.model.model_type} with input shape {input_shape}")

    # Wrap model in Lightning module
    lit_model = LitModel(model=net, cfg=cfg, num_classes=num_classes)

    # Setup WandB logger
    wandb_logger = WandbLogger(
        project=cfg.base.wandb.project,
        entity=cfg.base.wandb.entity,
        name=cfg.base.wandb.run_name if cfg.base.wandb.run_name else f"{cfg.base.job_id}_{cfg.model.model_type}",
        tags=cfg.base.wandb.tags,
        notes=cfg.base.wandb.notes,
        offline=not cfg.base.wandb.online,
        save_dir=cfg.train.model_path,
    )

    # Watch model parameters and gradients
    wandb_logger.watch(lit_model, log="all", log_freq=1000)

    # Log config to WandB
    if not cfg.base.debug:
        wandb_logger.experiment.config.update(cfg.model_dump(), allow_val_change=True)

    # Setup callbacks
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.train.model_path}/{cfg.base.job_id}",
        filename='{epoch:02d}-{val/acc:.4f}',
        monitor='val/acc',
        mode='max',
        save_top_k=3 if cfg.train.save_model else 0,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Early stopping (if enabled)
    if cfg.train.early_stopping.enabled:
        early_stop_callback = EarlyStopping(
            monitor=cfg.train.early_stopping.monitor,
            patience=cfg.train.early_stopping.patience,
            mode=cfg.train.early_stopping.mode,
            min_delta=cfg.train.early_stopping.min_delta,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        logger.info(f"Early stopping enabled: monitor={cfg.train.early_stopping.monitor}, patience={cfg.train.early_stopping.patience}")

    # Determine precision
    precision = "32"
    if cfg.train.dtype == torch.float16:
        precision = "16-mixed"
    elif cfg.train.dtype == torch.bfloat16:
        precision = "bf16-mixed"
    logger.info(f"Training precision: {precision}")

    # Create Lightning Trainer
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",  # Automatically selects GPU/CPU
        devices="auto",  # Use all available devices
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.train.lightning.log_every_n_steps,
        enable_progress_bar=cfg.train.lightning.enable_progress_bar,
        gradient_clip_val=cfg.train.lightning.gradient_clip_val,
        accumulate_grad_batches=cfg.train.lightning.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.train.lightning.check_val_every_n_epoch,
        num_sanity_val_steps=cfg.train.lightning.num_sanity_val_steps if not cfg.base.debug else 0,
        fast_dev_run=cfg.base.debug,  # Run 1 batch of train/val/test for debugging
        deterministic=True,  # For reproducibility
    )

    # Train the model
    logger.info(f"Starting training for {cfg.train.epochs} epochs...")
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.resume if args.resume else None,
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
    if not cfg.base.debug:
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
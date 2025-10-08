import argparse
import logging
import time

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim

from src import config
from src import data
from src import model
from src import io_funcs
from src import helpers
from src import train


def main():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('config_file', help='Config file path')
    parser.add_argument('--job_id', default='default', help='Job ID for checkpointing')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging')
    parser.add_argument('--override', action='append', help='Override config values using dotted notation. Examples: --override "train.epochs=10 train.learning_rate=0.001" or --override train.epochs=10 --override train.learning_rate=0.001')

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
    helpers.setup_seed(cfg.base.seed)

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
    train_dataloader, test_dataloader = data.get_dataloaders(cfg)

    # Get model parameters (accounting for transforms)
    input_shape = helpers.get_model_input_shape(cfg)
    num_classes = helpers.get_num_classes(cfg.data)
    logger.info(f"Input shape: {input_shape}, Num classes: {num_classes}")

    # Create model
    net = model.create_model(config=cfg, input_shape=input_shape, num_classes=num_classes)
    logger.info(f"Model: {cfg.model.model_type} with input shape {input_shape}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg.train.learning_rate)

    # Resume or start fresh
    start_batch = 0
    start_time = time.time()
    last_checkpoint_time = start_time

    if args.resume:
        checkpoint = io_funcs.load_checkpoint(args.job_id)
        if checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_batch = checkpoint['batch_count']
            # Adjust start time if we have elapsed time from checkpoint
            if 'elapsed_time' in checkpoint:
                start_time = time.time() - checkpoint['elapsed_time']
            logger.info(f"Resumed training from batch {start_batch}")
        else:
            logger.info("No checkpoint found, starting fresh")
    else:
        logger.info("Starting fresh training")

    # Training loop
    batch_count = start_batch
    max_batches = len(train_dataloader) * cfg.train.epochs

    logger.info(f"Training for {cfg.train.epochs} epochs ({max_batches} total batches)")
    logger.info(f"Checkpoint interval: {cfg.train.checkpoint_interval_minutes} minutes")

    for epoch in range(cfg.train.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{cfg.train.epochs}")

        # Train
        train_loss_per_sample, train_accuracy, batch_count, last_checkpoint_time = train.train_epoch(
            model=net,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            batch_count=batch_count,
            config=cfg,
            start_time=start_time,
            last_checkpoint_time=last_checkpoint_time
        )

        # Evaluate
        test_loss, test_accuracy = train.evaluate(
            model=net,
            dataloader=test_dataloader,
            criterion=criterion,
            config=cfg
        )

        logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss_per_sample:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

    # Save final checkpoint
    if cfg.train.save_model:
        final_elapsed_time = time.time() - start_time
        io_funcs.save_checkpoint(net, optimizer, batch_count, cfg, args.job_id, final_elapsed_time)
        logger.info(f"Final checkpoint saved for job {args.job_id}")

if __name__ == "__main__":
    main()
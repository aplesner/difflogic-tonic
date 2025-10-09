#!/usr/bin/env python3
"""
Main experiment runner using Hydra for configuration management.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path

from dataloaders import get_dataloader
from models import build_model
from trainer import Trainer
from utils.db_manager import DBManager


@hydra.main(config_path="configs", config_name="experiment", version_base=None)
def main(cfg: DictConfig):
    """
    Main experiment function.
    
    Args:
        cfg: Hydra configuration
    """
    print("="*60)
    print("DiffLUT Experiment Runner")
    print("="*60)
    
    # Convert config to dict for easier handling
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("="*60)
    
    # Initialize database manager
    db_path = Path(cfg.get('db_path', '../results/experiments.db'))
    db_manager = DBManager(str(db_path))
    print(f"Database: {db_path}")
    
    # Create experiment record
    experiment_name = cfg.get('experiment_name', None)
    experiment = db_manager.create_experiment(config_dict, name=experiment_name)
    print(f"Experiment ID: {experiment.id}")
    print(f"Experiment Name: {experiment.name}")
    
    # Setup dataloader
    print("\n" + "="*60)
    print("Setting up dataloader...")
    print("="*60)
    
    dataset_config = OmegaConf.to_container(cfg.dataloaders, resolve=True)
    dataloader = get_dataloader(cfg.dataloaders.name, dataset_config)
    
    # Prepare and setup data
    dataloader.prepare_data()
    train_loader, val_loader, test_loader = dataloader.setup()
    
    input_size = dataloader.get_input_size()
    num_classes = dataloader.get_num_classes()
    
    print(f"Dataset: {cfg.dataloaders.name}")
    print(f"Input size: {input_size}")
    print(f"Num classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Build model
    print("\n" + "="*60)
    print("Building model...")
    print("="*60)
    
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_model(model_config, input_size, num_classes)
    
    print(f"Model: {cfg.model.name}")
    
    # Fit encoder on training data
    print("\n" + "="*60)
    print("Fitting encoder on training data...")
    print("="*60)
    
    # Get a batch of training data for encoder fitting
    # We'll use all training data for proper statistics
    all_train_data = []
    all_train_labels = []
    for inputs, labels in train_loader:
        all_train_data.append(inputs)
        all_train_labels.append(labels)
    all_train_data = torch.cat(all_train_data, dim=0)
    
    # Fit encoder
    model.fit_encoder(all_train_data)
    
    # Setup trainer
    print("\n" + "="*60)
    print("Setting up trainer...")
    print("="*60)
    
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    
    # Add checkpoint directory to training config
    checkpoint_dir = Path(cfg.get('checkpoint_dir', '../results/checkpoints'))
    training_config['checkpoint_dir'] = str(checkpoint_dir)
    
    # Add gradient directory to training config
    gradient_dir = Path(cfg.get('gradient_dir', '../results/gradients'))
    training_config['gradient_dir'] = str(gradient_dir)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=training_config,
        db_manager=db_manager,
        experiment_id=experiment.id
    )
    
    # Train
    try:
        trainer.train()
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Training interrupted by user")
        print("="*60)
        db_manager.update_experiment_status(experiment.id, 'interrupted')
    except Exception as e:
        print("\n" + "="*60)
        print(f"Training failed with error: {e}")
        print("="*60)
        raise
    finally:
        # Close database connection
        db_manager.close()


if __name__ == "__main__":
    main()

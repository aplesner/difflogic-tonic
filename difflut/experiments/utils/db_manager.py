#!/usr/bin/env python3
"""
Database manager for experiment tracking using SQLAlchemy.
Manages experiments, measurement points, and metrics/checkpoints.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    ForeignKey, Text, JSON, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class Experiment(Base):
    """
    Experiment table stores all configuration information.
    Each experiment has a unique configuration.
    """
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Configuration as JSON
    config = Column(JSON, nullable=False)
    
    # Model configuration
    model_name = Column(String(100), nullable=False)
    node_type = Column(String(50))
    layer_type = Column(String(50))
    hidden_size = Column(Integer)
    num_layers = Column(Integer)
    n = Column(Integer)  # LUT inputs
    
    # Dataset configuration
    dataset_name = Column(String(100), nullable=False)
    batch_size = Column(Integer)
    
    # Training configuration
    epochs = Column(Integer)
    learning_rate = Column(Float)
    optimizer = Column(String(50))
    
    # Encoder configuration
    encoder_name = Column(String(50))
    encoder_params = Column(JSON)
    
    # Status
    status = Column(String(50), default='created')  # created, running, completed, failed
    
    # Device info
    device = Column(String(50))
    cuda_available = Column(Boolean)
    
    # Additional metadata
    notes = Column(Text)
    
    # Relationships
    measurement_points = relationship('MeasurementPoint', back_populates='experiment', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Experiment(id={self.id}, name='{self.name}', status='{self.status}')>"


class MeasurementPoint(Base):
    """
    Measurement point table stores time-based snapshots during training.
    Each point represents a specific moment (epoch, step) in training.
    """
    __tablename__ = 'measurement_points'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'), nullable=False)
    
    # Timing information
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    epoch = Column(Integer, nullable=False)
    step = Column(Integer)  # Optional: batch/step within epoch
    
    # Time tracking
    epoch_time = Column(Float)  # Time for this epoch in seconds
    total_time = Column(Float)  # Total time elapsed in seconds
    
    # Phase (train, validation, test)
    phase = Column(String(20), nullable=False)  # 'train', 'val', 'test'
    
    # Relationships
    experiment = relationship('Experiment', back_populates='measurement_points')
    metrics = relationship('Metric', back_populates='measurement_point', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<MeasurementPoint(id={self.id}, exp={self.experiment_id}, epoch={self.epoch}, phase='{self.phase}')>"


class Metric(Base):
    """
    Metric table stores metrics and checkpoints matched to measurement points.
    Each metric record is associated with a specific measurement point.
    """
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    measurement_point_id = Column(Integer, ForeignKey('measurement_points.id'), nullable=False)
    
    # Metric information
    metric_name = Column(String(100), nullable=False)  # e.g., 'loss', 'accuracy', 'f1'
    metric_value = Column(Float, nullable=False)
    
    # Optional: Additional metric metadata
    metric_metadata = Column(JSON)  # For storing additional context
    
    # Checkpoint information (optional)
    is_checkpoint = Column(Boolean, default=False)
    checkpoint_path = Column(String(500))  # Path to saved model checkpoint
    is_best = Column(Boolean, default=False)  # Flag for best model
    
    # Relationships
    measurement_point = relationship('MeasurementPoint', back_populates='metrics')
    
    def __repr__(self):
        return f"<Metric(id={self.id}, name='{self.metric_name}', value={self.metric_value:.4f})>"


class GradientSnapshot(Base):
    """
    Gradient snapshot table stores gradient information for validation samples.
    Each snapshot is associated with a measurement point.
    """
    __tablename__ = 'gradient_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    measurement_point_id = Column(Integer, ForeignKey('measurement_points.id'), nullable=False)
    
    # Gradient storage path
    gradient_path = Column(String(500), nullable=False)  # Path to saved gradient file
    
    # Sample information
    num_samples = Column(Integer)  # Number of samples gradients were computed for
    sample_indices = Column(JSON)  # Indices of samples used (optional)
    
    # Metadata
    gradient_metadata = Column(JSON)  # Additional information (layer names, shapes, etc.)
    
    # Relationships
    measurement_point = relationship('MeasurementPoint')
    
    def __repr__(self):
        return f"<GradientSnapshot(id={self.id}, mp_id={self.measurement_point_id}, path='{self.gradient_path}')>"


class DBManager:
    """
    Database manager for handling all database operations.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            # Default to results folder in project root
            results_dir = Path(__file__).parent.parent.parent / 'results'
            results_dir.mkdir(exist_ok=True)
            db_path = str(results_dir / 'experiments.db')
        else:
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        
        # Create engine
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            connect_args={'check_same_thread': False},
            poolclass=StaticPool
        )
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def create_experiment(self, config: Dict[str, Any], name: Optional[str] = None) -> Experiment:
        """
        Create a new experiment record.
        
        Args:
            config: Complete experiment configuration
            name: Optional experiment name (auto-generated if None)
        
        Returns:
            Created Experiment object
        """
        session = self.get_session()
        
        try:
            # Generate name if not provided
            if name is None:
                name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Extract configuration details
            model_config = config.get('model', {})
            dataset_config = config.get('dataloaders', config.get('dataset', {}))  # Support both keys
            training_config = config.get('training', {})
            
            # Get encoder config from model or dataset
            model_params = model_config.get('params', {})
            encoder_config = model_params.get('encoder', dataset_config.get('encoder', {}))
            
            # Get layer config - support both flattened and nested config
            if 'layer_type' in model_params and 'node_type' in model_params:
                # Flattened config
                node_type = model_params.get('node_type')
                layer_type = model_params.get('layer_type')
                hidden_sizes = model_params.get('hidden_sizes', [None])
                hidden_size = hidden_sizes[0] if hidden_sizes else None
                num_layers = len(hidden_sizes) if hidden_sizes else 0
                n = model_params.get('num_inputs')
            else:
                # Nested config (backward compatibility)
                layers = model_params.get('layers', model_config.get('layers', [{}]))
                first_layer = layers[0] if layers else {}
                node_config = first_layer.get('node', {})
                node_type = node_config.get('type')
                layer_type = first_layer.get('type')
                hidden_size = first_layer.get('hidden_sizes', [None])[0]
                num_layers = len(layers)
                n = node_config.get('parameters', {}).get('num_inputs')
            
            experiment = Experiment(
                name=name,
                config=config,
                model_name=model_config.get('name', 'unknown'),
                node_type=node_type,
                layer_type=layer_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                n=n,
                dataset_name=dataset_config.get('name', 'unknown'),
                batch_size=training_config.get('batch_size'),
                epochs=training_config.get('epochs'),
                learning_rate=training_config.get('lr'),
                optimizer=training_config.get('optimizer'),
                encoder_name=encoder_config.get('name'),
                encoder_params=encoder_config.get('parameters'),
                status='created'
            )
            
            session.add(experiment)
            session.commit()
            session.refresh(experiment)
            
            return experiment
        finally:
            session.close()
    
    def update_experiment_status(self, experiment_id: int, status: str, device: Optional[str] = None):
        """
        Update experiment status.
        
        Args:
            experiment_id: ID of the experiment
            status: New status ('running', 'completed', 'failed')
            device: Optional device information
        """
        session = self.get_session()
        
        try:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if experiment:
                experiment.status = status
                if device:
                    experiment.device = device
                session.commit()
        finally:
            session.close()
    
    def create_measurement_point(
        self, 
        experiment_id: int, 
        epoch: int, 
        phase: str,
        step: Optional[int] = None,
        epoch_time: Optional[float] = None,
        total_time: Optional[float] = None
    ) -> MeasurementPoint:
        """
        Create a new measurement point.
        
        Args:
            experiment_id: ID of the experiment
            epoch: Current epoch number
            phase: Phase of training ('train', 'val', 'test')
            step: Optional step/batch number
            epoch_time: Time taken for this epoch
            total_time: Total elapsed time
        
        Returns:
            Created MeasurementPoint object
        """
        session = self.get_session()
        
        try:
            measurement_point = MeasurementPoint(
                experiment_id=experiment_id,
                epoch=epoch,
                step=step,
                phase=phase,
                epoch_time=epoch_time,
                total_time=total_time
            )
            
            session.add(measurement_point)
            session.commit()
            session.refresh(measurement_point)
            
            return measurement_point
        finally:
            session.close()
    
    def add_metrics(
        self, 
        measurement_point_id: int, 
        metrics: Dict[str, float],
        checkpoint_path: Optional[str] = None,
        is_best: bool = False
    ):
        """
        Add metrics to a measurement point.
        
        Args:
            measurement_point_id: ID of the measurement point
            metrics: Dictionary of metric names and values
            checkpoint_path: Optional path to saved checkpoint
            is_best: Whether this is the best model so far
        """
        session = self.get_session()
        
        try:
            for metric_name, metric_value in metrics.items():
                metric = Metric(
                    measurement_point_id=measurement_point_id,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    is_checkpoint=checkpoint_path is not None,
                    checkpoint_path=checkpoint_path,
                    is_best=is_best
                )
                session.add(metric)
            
            session.commit()
        finally:
            session.close()
    
    def get_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Get experiment by ID."""
        session = self.get_session()
        try:
            return session.query(Experiment).filter_by(id=experiment_id).first()
        finally:
            session.close()
    
    def get_experiment_metrics(self, experiment_id: int, phase: Optional[str] = None) -> List[Dict]:
        """
        Get all metrics for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            phase: Optional phase filter ('train', 'val', 'test')
        
        Returns:
            List of metric dictionaries
        """
        session = self.get_session()
        
        try:
            query = session.query(MeasurementPoint, Metric).join(Metric)
            query = query.filter(MeasurementPoint.experiment_id == experiment_id)
            
            if phase:
                query = query.filter(MeasurementPoint.phase == phase)
            
            results = []
            for mp, metric in query.all():
                results.append({
                    'epoch': mp.epoch,
                    'step': mp.step,
                    'phase': mp.phase,
                    'timestamp': mp.timestamp,
                    'metric_name': metric.metric_name,
                    'metric_value': metric.metric_value,
                    'is_best': metric.is_best
                })
            
            return results
        finally:
            session.close()
    
    def get_best_checkpoint(self, experiment_id: int, metric_name: str = 'accuracy') -> Optional[str]:
        """
        Get path to best checkpoint for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            metric_name: Metric to use for determining best
        
        Returns:
            Path to best checkpoint or None
        """
        session = self.get_session()
        
        try:
            query = session.query(Metric).join(MeasurementPoint)
            query = query.filter(
                MeasurementPoint.experiment_id == experiment_id,
                Metric.metric_name == metric_name,
                Metric.is_best == True
            )
            
            metric = query.first()
            return metric.checkpoint_path if metric else None
        finally:
            session.close()
    
    def list_experiments(self, limit: int = 100) -> List[Experiment]:
        """
        List recent experiments.
        
        Args:
            limit: Maximum number of experiments to return
        
        Returns:
            List of Experiment objects
        """
        session = self.get_session()
        
        try:
            return session.query(Experiment).order_by(
                Experiment.created_at.desc()
            ).limit(limit).all()
        finally:
            session.close()
    
    def add_gradient_snapshot(
        self,
        measurement_point_id: int,
        gradient_path: str,
        num_samples: Optional[int] = None,
        sample_indices: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add gradient snapshot information.
        
        Args:
            measurement_point_id: ID of the measurement point
            gradient_path: Path to saved gradient file
            num_samples: Number of samples used for gradient computation
            sample_indices: Indices of samples used
            metadata: Additional metadata (layer names, shapes, etc.)
        """
        session = self.get_session()
        
        try:
            snapshot = GradientSnapshot(
                measurement_point_id=measurement_point_id,
                gradient_path=gradient_path,
                num_samples=num_samples,
                sample_indices=sample_indices,
                gradient_metadata=metadata
            )
            session.add(snapshot)
            session.commit()
        finally:
            session.close()
    
    def get_gradient_snapshots(self, experiment_id: int) -> List[Dict]:
        """
        Get all gradient snapshots for an experiment.
        
        Args:
            experiment_id: ID of the experiment
        
        Returns:
            List of gradient snapshot dictionaries
        """
        session = self.get_session()
        
        try:
            query = session.query(MeasurementPoint, GradientSnapshot).join(GradientSnapshot)
            query = query.filter(MeasurementPoint.experiment_id == experiment_id)
            
            results = []
            for mp, snapshot in query.all():
                results.append({
                    'epoch': mp.epoch,
                    'phase': mp.phase,
                    'gradient_path': snapshot.gradient_path,
                    'num_samples': snapshot.num_samples,
                    'sample_indices': snapshot.sample_indices,
                    'metadata': snapshot.gradient_metadata
                })
            
            return results
        finally:
            session.close()
    
    def close(self):
        """Close database connection."""
        self.engine.dispose()

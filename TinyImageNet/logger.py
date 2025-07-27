import logging
import os
import sys
from datetime import datetime


class TrainingLogger:
    def __init__(self, log_dir, log_name='training'):
        """
        Initialize the training logger
        
        Args:
            log_dir: Directory to save log files
            log_name: Name of the log file (without extension)
        """
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create timestamp for unique log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{log_name}_{timestamp}.log')
        
        # Set up logger
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Store log file path
        self.log_file = log_file
        
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
        
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
        
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
        
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
        
    def log_epoch(self, epoch, metrics):
        """Log epoch metrics"""
        metric_str = ', '.join([f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' 
                               for k, v in metrics.items()])
        self.info(f'Epoch {epoch} - {metric_str}')
        
    def log_round(self, round_num, task_id, metrics):
        """Log federated round metrics"""
        metric_str = ', '.join([f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' 
                               for k, v in metrics.items()])
        self.info(f'Round {round_num}, Task {task_id} - {metric_str}')
        
    def log_client(self, client_id, message):
        """Log client-specific message"""
        self.info(f'Client {client_id} - {message}')
        
    def log_config(self, config):
        """Log configuration parameters"""
        self.info('='*50)
        self.info('Configuration:')
        for key, value in vars(config).items():
            self.info(f'{key}: {value}')
        self.info('='*50)
        
    def close(self):
        """Close all handlers"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
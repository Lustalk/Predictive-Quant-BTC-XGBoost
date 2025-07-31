"""
Logging utilities for the trading system.
Provides structured logging with file rotation and different output formats.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger as loguru_logger


class TradingLogger:
    """Enhanced logging system for trading application."""
    
    def __init__(self, 
                 name: str = "trading_system",
                 log_level: str = "INFO",
                 log_dir: str = "logs",
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True):
        """
        Initialize the trading logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files
            enable_file_logging: Whether to enable file logging
            enable_console_logging: Whether to enable console logging
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self._setup_standard_logger()
        self._setup_loguru_logger()
    
    def _setup_standard_logger(self) -> None:
        """Setup standard Python logger."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.enable_file_logging:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.name}.log",
                maxBytes=100 * 1024 * 1024,  # 100MB
                backupCount=5
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
    
    def _setup_loguru_logger(self) -> None:
        """Setup loguru logger for enhanced logging features."""
        # Remove default loguru handler
        loguru_logger.remove()
        
        # Add console handler
        if self.enable_console_logging:
            loguru_logger.add(
                sys.stdout,
                level=self.log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                      "<level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                      "<level>{message}</level>",
                colorize=True
            )
        
        # Add file handler
        if self.enable_file_logging:
            loguru_logger.add(
                self.log_dir / f"{self.name}_loguru.log",
                level=self.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="100 MB",
                retention="10 days",
                compression="zip"
            )
    
    def get_logger(self) -> logging.Logger:
        """Get the standard Python logger."""
        return self.logger
    
    def get_loguru_logger(self):
        """Get the loguru logger."""
        return loguru_logger
    
    def log_trade(self, 
                  action: str, 
                  symbol: str, 
                  price: float, 
                  quantity: float, 
                  timestamp: str,
                  additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Log trading activities with structured format.
        
        Args:
            action: Trade action (BUY, SELL, HOLD)
            symbol: Trading symbol
            price: Execution price
            quantity: Trade quantity
            timestamp: Trade timestamp
            additional_info: Additional trade information
        """
        trade_info = {
            "action": action,
            "symbol": symbol,
            "price": price,
            "quantity": quantity,
            "timestamp": timestamp,
            "value": price * quantity
        }
        
        if additional_info:
            trade_info.update(additional_info)
        
        # Log to dedicated trade log
        trade_logger = logging.getLogger(f"{self.name}.trades")
        trade_logger.info(f"TRADE: {trade_info}")
        
        # Also log to loguru with structured data
        loguru_logger.bind(**trade_info).info("Trade executed")
    
    def log_performance(self, 
                       metrics: Dict[str, float],
                       period: str = "daily") -> None:
        """
        Log performance metrics.
        
        Args:
            metrics: Performance metrics dictionary
            period: Performance period (daily, weekly, monthly)
        """
        perf_logger = logging.getLogger(f"{self.name}.performance")
        perf_logger.info(f"PERFORMANCE_{period.upper()}: {metrics}")
        
        loguru_logger.bind(period=period, **metrics).info("Performance metrics")
    
    def log_system_health(self, 
                         memory_usage: float,
                         cpu_usage: float,
                         active_connections: int) -> None:
        """
        Log system health metrics.
        
        Args:
            memory_usage: Memory usage in MB
            cpu_usage: CPU usage percentage
            active_connections: Number of active connections
        """
        health_metrics = {
            "memory_usage_mb": memory_usage,
            "cpu_usage_percent": cpu_usage,
            "active_connections": active_connections
        }
        
        health_logger = logging.getLogger(f"{self.name}.health")
        health_logger.info(f"SYSTEM_HEALTH: {health_metrics}")
        
        loguru_logger.bind(**health_metrics).info("System health check")


# Global logger instance
_global_logger: Optional[TradingLogger] = None


def setup_logging(config: Optional[Dict[str, Any]] = None) -> TradingLogger:
    """
    Setup global logging configuration.
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured TradingLogger instance
    """
    global _global_logger
    
    if config is None:
        config = {
            "level": "INFO",
            "log_dir": "logs",
            "enable_file_logging": True,
            "enable_console_logging": True
        }
    
    _global_logger = TradingLogger(
        name="btc_trading",
        log_level=config.get("level", "INFO"),
        log_dir=config.get("log_dir", "logs"),
        enable_file_logging=config.get("enable_file_logging", True),
        enable_console_logging=config.get("enable_console_logging", True)
    )
    
    return _global_logger


def get_logger() -> TradingLogger:
    """
    Get the global logger instance.
    
    Returns:
        Global TradingLogger instance
        
    Raises:
        RuntimeError: If logging hasn't been setup
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = setup_logging()
    
    return _global_logger


def log_trade(action: str, symbol: str, price: float, quantity: float, **kwargs) -> None:
    """Convenience function for logging trades."""
    logger = get_logger()
    logger.log_trade(action, symbol, price, quantity, **kwargs)


def log_performance(metrics: Dict[str, float], period: str = "daily") -> None:
    """Convenience function for logging performance."""
    logger = get_logger()
    logger.log_performance(metrics, period)
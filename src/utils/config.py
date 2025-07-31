"""
Configuration management utilities.
Handles loading and validation of configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    backend: str = "duckdb"
    database_url: str = "data/trading_data.db"


class ExchangeConfig(BaseModel):
    """Exchange API configuration."""
    name: str = "binance"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = True
    
    class Config:
        env_prefix = "EXCHANGE_"


class ModelConfig(BaseModel):
    """Machine learning model configuration."""
    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42


class TradingConfig(BaseSettings):
    """Main trading configuration class."""
    
    # Application settings
    app_name: str = Field("BTC Trading Strategy", env="APP_NAME")
    app_version: str = "0.1.0"
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")
    
    # Database settings
    database: DatabaseConfig = DatabaseConfig()
    
    # Exchange settings
    exchange: ExchangeConfig = ExchangeConfig()
    
    # Model settings
    xgboost_params: ModelConfig = ModelConfig()
    lightgbm_params: ModelConfig = ModelConfig()
    
    # Paths
    data_dir: Path = Path("data")
    logs_dir: Path = Path("logs")
    models_dir: Path = Path("models")
    config_dir: Path = Path("config")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ConfigManager:
    """Configuration manager for loading and validating settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or "config/settings.yaml"
        self._config_data: Optional[Dict[str, Any]] = None
        self._trading_config: Optional[TradingConfig] = None
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration data.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            yaml.YAMLError: If YAML parsing fails.
        """
        if self._config_data is None:
            config_file = Path(self.config_path)
            
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
            with open(config_file, 'r', encoding='utf-8') as file:
                # Replace environment variables in the YAML content
                content = file.read()
                content = self._replace_env_variables(content)
                self._config_data = yaml.safe_load(content)
        
        return self._config_data
    
    def get_trading_config(self) -> TradingConfig:
        """
        Get validated trading configuration.
        
        Returns:
            TradingConfig instance with validated settings.
        """
        if self._trading_config is None:
            # Load base configuration
            config_data = self.load_config()
            
            # Create TradingConfig with environment variable support
            self._trading_config = TradingConfig()
            
            # Update with YAML values where available
            if 'app' in config_data:
                app_config = config_data['app']
                self._trading_config.app_name = app_config.get('name', self._trading_config.app_name)
                self._trading_config.app_version = app_config.get('version', self._trading_config.app_version)
                self._trading_config.environment = app_config.get('environment', self._trading_config.environment)
                self._trading_config.debug = app_config.get('debug', self._trading_config.debug)
        
        return self._trading_config
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get specific configuration section.
        
        Args:
            section_name: Name of the configuration section.
            
        Returns:
            Dictionary containing section configuration.
            
        Raises:
            KeyError: If section doesn't exist.
        """
        config_data = self.load_config()
        
        if section_name not in config_data:
            raise KeyError(f"Configuration section '{section_name}' not found")
        
        return config_data[section_name]
    
    def _replace_env_variables(self, content: str) -> str:
        """
        Replace environment variable placeholders in configuration content.
        
        Args:
            content: YAML content with ${VAR_NAME} placeholders.
            
        Returns:
            Content with environment variables replaced.
        """
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        
        # Replace ${VAR_NAME} patterns
        return re.sub(r'\$\{([^}]+)\}', replace_var, content)
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        config = self.get_trading_config()
        
        directories = [
            config.data_dir,
            config.logs_dir,
            config.models_dir,
            config.config_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise.
        """
        try:
            # Load and validate trading config
            trading_config = self.get_trading_config()
            
            # Validate required sections exist
            config_data = self.load_config()
            required_sections = ['data', 'models', 'strategy', 'backtesting']
            
            for section in required_sections:
                if section not in config_data:
                    print(f"Missing required configuration section: {section}")
                    return False
            
            # Validate model parameters
            models_config = config_data.get('models', {})
            if 'xgboost' not in models_config or 'lightgbm' not in models_config:
                print("Missing required model configurations")
                return False
            
            print("Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> TradingConfig:
    """Get the global trading configuration."""
    return config_manager.get_trading_config()


def get_config_section(section_name: str) -> Dict[str, Any]:
    """Get a specific configuration section."""
    return config_manager.get_section(section_name)
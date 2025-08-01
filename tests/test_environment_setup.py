"""
Test environment setup and project structure.
Tests from Checkpoint 1.1 as specified in the roadmap.
"""

import pytest
import importlib
import sys
from pathlib import Path
import logging
import yaml
import docker
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config import ConfigManager, TradingConfig
from src.utils.logging import TradingLogger, setup_logging


class TestEnvironmentSetup:
    """Test that all dependencies are correctly installed."""

    def test_core_data_science_imports(self):
        """Test that core data science libraries can be imported."""
        core_libraries = [
            "pandas",
            "numpy",
            "scipy",
            "sklearn",
            "matplotlib",
            "seaborn",
            "plotly",
        ]

        for library in core_libraries:
            try:
                importlib.import_module(library)
            except ImportError as e:
                pytest.fail(f"Failed to import core library {library}: {e}")

    def test_ml_libraries_import(self):
        """Test that machine learning libraries can be imported."""
        ml_libraries = ["xgboost", "lightgbm"]

        for library in ml_libraries:
            try:
                importlib.import_module(library)
            except ImportError as e:
                pytest.fail(f"Failed to import ML library {library}: {e}")

    def test_financial_libraries_import(self):
        """Test that financial libraries can be imported."""
        financial_libraries = [
            "ccxt"
        ]  # We'll skip binance for now as it needs API keys

        for library in financial_libraries:
            try:
                importlib.import_module(library)
            except ImportError as e:
                pytest.fail(f"Failed to import financial library {library}: {e}")

    def test_database_libraries_import(self):
        """Test that database libraries can be imported."""
        db_libraries = ["sqlalchemy", "duckdb"]

        for library in db_libraries:
            try:
                importlib.import_module(library)
            except ImportError as e:
                pytest.fail(f"Failed to import database library {library}: {e}")

    def test_async_libraries_import(self):
        """Test that async libraries can be imported."""
        async_libraries = ["aiohttp", "websockets"]

        for library in async_libraries:
            try:
                importlib.import_module(library)
            except ImportError as e:
                pytest.fail(f"Failed to import async library {library}: {e}")

    def test_testing_libraries_import(self):
        """Test that testing libraries can be imported."""
        testing_libraries = ["pytest"]

        for library in testing_libraries:
            try:
                importlib.import_module(library)
            except ImportError as e:
                pytest.fail(f"Failed to import testing library {library}: {e}")

    @pytest.mark.skip(reason="Docker daemon may not be available in CI")
    def test_docker_compose_up(self):
        """Test that Docker Compose can start services."""
        try:
            client = docker.from_env()
            # Just test that Docker client can connect
            client.ping()
            assert True, "Docker client connected successfully"
        except Exception as e:
            pytest.fail(f"Docker test failed: {e}")

    def test_logging_configuration_valid(self):
        """Test that logging configuration is valid."""
        try:
            # Test basic logging setup
            logger = setup_logging(
                {
                    "level": "INFO",
                    "log_dir": "logs",
                    "enable_file_logging": False,  # Disable file logging for tests
                    "enable_console_logging": True,
                }
            )

            assert isinstance(logger, TradingLogger)
            assert logger.logger.level == logging.INFO

            # Test logging functionality
            std_logger = logger.get_logger()
            std_logger.info("Test log message")

            # Test structured logging
            logger.log_trade("BUY", "BTCUSDT", 50000.0, 0.1, "2023-01-01T00:00:00Z")

        except Exception as e:
            pytest.fail(f"Logging configuration test failed: {e}")


class TestProjectStructure:
    """Verify project structure follows standards."""

    def test_all_required_directories_exist(self):
        """Test that all required directories exist."""
        project_root = Path(__file__).parent.parent

        required_directories = [
            ".github/workflows",
            "config",
            "data",
            "src/data",
            "src/features",
            "src/models",
            "src/strategy",
            "src/backtesting",
            "src/utils",
            "tests",
            "notebooks",
        ]

        for directory in required_directories:
            dir_path = project_root / directory
            assert dir_path.exists(), f"Required directory missing: {directory}"
            assert dir_path.is_dir(), f"Path exists but is not a directory: {directory}"

    def test_python_package_structure(self):
        """Test that Python packages have __init__.py files."""
        project_root = Path(__file__).parent.parent

        python_packages = [
            "src",
            "src/data",
            "src/features",
            "src/models",
            "src/strategy",
            "src/backtesting",
            "src/utils",
            "tests",
            "config",
        ]

        for package in python_packages:
            init_file = project_root / package / "__init__.py"
            assert init_file.exists(), f"Missing __init__.py in package: {package}"

    def test_configuration_files_valid(self):
        """Test that configuration files are valid."""
        project_root = Path(__file__).parent.parent

        # Test main configuration file
        config_file = project_root / "config" / "settings.yaml"
        assert config_file.exists(), "Main configuration file missing"

        # Test YAML validity
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
            assert isinstance(
                config_data, dict
            ), "Configuration is not a valid dictionary"

        # Test required sections
        required_sections = ["app", "data", "models", "strategy", "backtesting"]
        for section in required_sections:
            assert (
                section in config_data
            ), f"Missing required configuration section: {section}"

    def test_docker_files_exist(self):
        """Test that Docker files exist and are valid."""
        project_root = Path(__file__).parent.parent

        docker_files = ["Dockerfile", "docker-compose.yml"]

        for docker_file in docker_files:
            file_path = project_root / docker_file
            assert file_path.exists(), f"Missing Docker file: {docker_file}"
            assert file_path.stat().st_size > 0, f"Docker file is empty: {docker_file}"

    def test_requirements_files_exist(self):
        """Test that requirements files exist."""
        project_root = Path(__file__).parent.parent

        requirement_files = ["requirements.txt", "pyproject.toml"]

        for req_file in requirement_files:
            file_path = project_root / req_file
            assert file_path.exists(), f"Missing requirements file: {req_file}"
            assert (
                file_path.stat().st_size > 0
            ), f"Requirements file is empty: {req_file}"


class TestConfigurationManagement:
    """Test configuration management system."""

    def test_config_manager_initialization(self):
        """Test ConfigManager can be initialized."""
        config_manager = ConfigManager()
        assert config_manager is not None

        # Test with custom path
        config_manager_custom = ConfigManager("config/settings.yaml")
        assert config_manager_custom.config_path == "config/settings.yaml"

    def test_config_loading(self):
        """Test configuration loading."""
        config_manager = ConfigManager("config/settings.yaml")

        try:
            config_data = config_manager.load_config()
            assert isinstance(config_data, dict)
            assert len(config_data) > 0
        except FileNotFoundError:
            pytest.skip("Configuration file not found - may be normal in isolated test")

    def test_trading_config_creation(self):
        """Test TradingConfig creation."""
        # Test with default values
        trading_config = TradingConfig()

        assert trading_config.app_name == "BTC Trading Strategy"
        assert trading_config.app_version == "0.1.0"
        assert trading_config.environment in ["development", "staging", "production"]

    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager("config/settings.yaml")

        try:
            is_valid = config_manager.validate_config()
            # If config file exists, it should be valid
            # If it doesn't exist, skip the test
            assert is_valid or not Path("config/settings.yaml").exists()
        except FileNotFoundError:
            pytest.skip("Configuration file not found - may be normal in isolated test")


class TestMemoryAndPerformance:
    """Test memory and performance constraints."""

    def test_import_memory_usage(self):
        """Test that imports don't consume excessive memory."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Import major libraries
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        import lightgbm as lgb

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use more than 500MB just for imports
        assert (
            memory_increase < 500
        ), f"Imports used too much memory: {memory_increase:.2f}MB"

    def test_basic_operations_performance(self):
        """Test that basic operations perform within acceptable time."""
        import time
        import pandas as pd
        import numpy as np

        # Test pandas DataFrame creation
        start_time = time.time()
        df = pd.DataFrame(np.random.randn(10000, 10))
        df_creation_time = time.time() - start_time

        assert (
            df_creation_time < 1.0
        ), f"DataFrame creation too slow: {df_creation_time:.3f}s"

        # Test basic calculations
        start_time = time.time()
        df["sma_20"] = df[0].rolling(20).mean()
        calculation_time = time.time() - start_time

        assert (
            calculation_time < 1.0
        ), f"Basic calculations too slow: {calculation_time:.3f}s"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

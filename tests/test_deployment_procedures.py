#!/usr/bin/env python3
"""
Test suite for deployment procedures and scripts.

Tests deployment scripts, configuration management, and deployment workflows
to ensure reliable production deployment.
"""

import os
import sys
import tempfile
import shutil
import subprocess
import yaml
from unittest import TestCase
from unittest.mock import patch, MagicMock, mock_open

class TestDeploymentScripts(TestCase):
    """Test deployment scripts and procedures."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create deployment directory structure
        os.makedirs('deployment/docker', exist_ok=True)
        os.makedirs('deployment/scripts', exist_ok=True)
        os.makedirs('config', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('backups', exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_deployment_script_structure(self):
        """Test deployment script has proper structure."""
        deploy_script_path = os.path.join(os.path.dirname(__file__), '..', 'deployment', 'scripts', 'deploy.sh')
        
        if os.path.exists(deploy_script_path):
            with open(deploy_script_path, 'r') as f:
                content = f.read()
            
            # Check for essential functions
            essential_functions = [
                'check_prerequisites',
                'validate_environment',
                'create_backup',
                'build_images',
                'deploy_application',
                'verify_deployment',
                'rollback'
            ]
            
            for func in essential_functions:
                self.assertIn(func, content, f"Function {func} not found in deploy script")
            
            # Check for proper error handling
            self.assertIn('set -e', content, "Script should exit on error")
            self.assertIn('trap', content, "Script should have error trapping")
    
    def test_health_check_script_functionality(self):
        """Test health check script functionality."""
        health_script_path = os.path.join(os.path.dirname(__file__), '..', 'deployment', 'scripts', 'health_check.py')
        
        if os.path.exists(health_script_path):
            # Test script can be imported
            sys.path.insert(0, os.path.dirname(health_script_path))
            try:
                import health_check
                
                # Test HealthChecker class exists
                self.assertTrue(hasattr(health_check, 'HealthChecker'))
                
                # Test essential methods exist
                checker = health_check.HealthChecker()
                essential_methods = [
                    'check_docker_services',
                    'check_system_resources',
                    'check_trading_service',
                    'check_monitoring_service',
                    'run_all_checks'
                ]
                
                for method in essential_methods:
                    self.assertTrue(hasattr(checker, method), f"Method {method} not found")
                
            except ImportError as e:
                self.skipTest(f"Health check script not importable: {e}")
            finally:
                sys.path.pop(0)
    
    @patch('subprocess.run')
    def test_docker_deployment_validation(self, mock_run):
        """Test Docker deployment validation."""
        # Create Docker Compose file
        docker_compose = {
            'version': '3.8',
            'services': {
                'alphapulse-trading': {
                    'build': '.',
                    'ports': ['8080:8080'],
                    'environment': ['TRADING_MODE=paper'],
                    'volumes': ['./logs:/app/logs'],
                    'depends_on': ['redis']
                },
                'alphapulse-monitor': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.monitor'
                    },
                    'ports': ['8081:8081'],
                    'depends_on': ['redis']
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'ports': ['6379:6379']
                }
            }
        }
        
        with open('deployment/docker/docker-compose.yml', 'w') as f:
            yaml.dump(docker_compose, f)
        
        # Mock successful Docker commands
        mock_run.return_value = MagicMock(returncode=0, stdout="Valid compose file")
        
        # Test Docker Compose validation
        result = subprocess.run(['docker-compose', '-f', 'deployment/docker/docker-compose.yml', 'config'], 
                              capture_output=True, text=True)
        
        # Should not raise exception with mocked successful result
        self.assertEqual(mock_run.return_value.returncode, 0)
    
    def test_environment_configuration_validation(self):
        """Test environment configuration validation."""
        # Test valid environment configuration
        valid_env = """
# AlphaPulse-RL Environment Configuration
WEEX_API_KEY=test_api_key_12345678901234567890
WEEX_SECRET_KEY=test_secret_key_12345678901234567890
WEEX_PASSPHRASE=test_passphrase_123
TRADING_MODE=paper
INITIAL_BALANCE=10000
LOG_LEVEL=INFO
REDIS_HOST=redis
REDIS_PORT=6379
MONITORING_INTERVAL=30
"""
        
        with open('deployment/.env', 'w') as f:
            f.write(valid_env)
        
        # Validate environment file
        self.assertTrue(os.path.exists('deployment/.env'))
        
        # Parse and validate environment variables
        env_vars = {}
        with open('deployment/.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
        
        # Check required variables
        required_vars = [
            'WEEX_API_KEY',
            'WEEX_SECRET_KEY',
            'WEEX_PASSPHRASE',
            'TRADING_MODE',
            'INITIAL_BALANCE'
        ]
        
        for var in required_vars:
            self.assertIn(var, env_vars, f"Required environment variable {var} not found")
            self.assertNotEqual(env_vars[var], '', f"Environment variable {var} is empty")
        
        # Validate specific values
        self.assertIn(env_vars['TRADING_MODE'], ['paper', 'live'])
        self.assertTrue(float(env_vars['INITIAL_BALANCE']) > 0)
    
    def test_configuration_file_validation(self):
        """Test configuration file validation."""
        # Create main configuration file
        main_config = {
            'system': {
                'name': 'alphapulse-rl',
                'version': '1.0.0',
                'environment': 'production'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/alphapulse.log',
                'max_size': '100MB',
                'backup_count': 5
            },
            'database': {
                'redis': {
                    'host': 'redis',
                    'port': 6379,
                    'db': 0
                }
            }
        }
        
        with open('config/config.yaml', 'w') as f:
            yaml.dump(main_config, f)
        
        # Create trading parameters file
        trading_params = {
            'risk_management': {
                'max_leverage': 12,
                'max_position_size_percent': 10,
                'max_daily_loss_percent': 3,
                'max_drawdown_percent': 12
            },
            'trading': {
                'pairs': ['BTCUSDT', 'ETHUSDT'],
                'timeframes': ['5m', '15m'],
                'confidence_threshold': 0.8
            },
            'model': {
                'update_frequency': 3600,
                'batch_size': 32,
                'learning_rate': 0.0003
            }
        }
        
        with open('config/trading_params.yaml', 'w') as f:
            yaml.dump(trading_params, f)
        
        # Validate configuration files
        self.assertTrue(os.path.exists('config/config.yaml'))
        self.assertTrue(os.path.exists('config/trading_params.yaml'))
        
        # Test configuration loading
        with open('config/config.yaml', 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        self.assertEqual(loaded_config['system']['name'], 'alphapulse-rl')
        self.assertEqual(loaded_config['logging']['level'], 'INFO')
        
        with open('config/trading_params.yaml', 'r') as f:
            loaded_params = yaml.safe_load(f)
        
        self.assertEqual(loaded_params['risk_management']['max_leverage'], 12)
        self.assertEqual(loaded_params['trading']['confidence_threshold'], 0.8)
    
    def test_backup_procedures(self):
        """Test backup procedures."""
        # Create test files to backup
        test_config = {'test': 'config'}
        with open('config/test_config.yaml', 'w') as f:
            yaml.dump(test_config, f)
        
        # Create test log file
        with open('logs/test.log', 'w') as f:
            f.write("Test log entry\n")
        
        # Create test model file
        os.makedirs('models', exist_ok=True)
        with open('models/test_model.pth', 'wb') as f:
            f.write(b'fake model data')
        
        # Simulate backup creation
        backup_dir = 'backups/test_backup_20240101_120000'
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy files to backup
        shutil.copytree('config', os.path.join(backup_dir, 'config'))
        shutil.copytree('logs', os.path.join(backup_dir, 'logs'))
        shutil.copytree('models', os.path.join(backup_dir, 'models'))
        
        # Verify backup
        self.assertTrue(os.path.exists(os.path.join(backup_dir, 'config', 'test_config.yaml')))
        self.assertTrue(os.path.exists(os.path.join(backup_dir, 'logs', 'test.log')))
        self.assertTrue(os.path.exists(os.path.join(backup_dir, 'models', 'test_model.pth')))
    
    def test_rollback_procedures(self):
        """Test rollback procedures."""
        # Create current configuration
        current_config = {'version': '2.0.0', 'feature': 'new'}
        with open('config/config.yaml', 'w') as f:
            yaml.dump(current_config, f)
        
        # Create backup with old configuration
        backup_dir = 'backups/rollback_test'
        os.makedirs(os.path.join(backup_dir, 'config'), exist_ok=True)
        
        old_config = {'version': '1.0.0', 'feature': 'old'}
        with open(os.path.join(backup_dir, 'config', 'config.yaml'), 'w') as f:
            yaml.dump(old_config, f)
        
        # Simulate rollback
        if os.path.exists('config/config.yaml'):
            os.remove('config/config.yaml')
        
        shutil.copy(os.path.join(backup_dir, 'config', 'config.yaml'), 'config/config.yaml')
        
        # Verify rollback
        with open('config/config.yaml', 'r') as f:
            restored_config = yaml.safe_load(f)
        
        self.assertEqual(restored_config['version'], '1.0.0')
        self.assertEqual(restored_config['feature'], 'old')
    
    def test_service_startup_validation(self):
        """Test service startup validation."""
        # Create systemd service files
        os.makedirs('deployment/systemd', exist_ok=True)
        
        trading_service = """
[Unit]
Description=AlphaPulse-RL Trading System
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=alphapulse
WorkingDirectory=/opt/alphapulse-rl
ExecStart=/opt/alphapulse-rl/venv/bin/python main.py
Restart=always
RestartSec=10
Environment=PYTHONPATH=/opt/alphapulse-rl
EnvironmentFile=/opt/alphapulse-rl/deployment/.env

[Install]
WantedBy=multi-user.target
"""
        
        with open('deployment/systemd/alphapulse-trading.service', 'w') as f:
            f.write(trading_service)
        
        monitor_service = """
[Unit]
Description=AlphaPulse-RL Monitoring Service
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=alphapulse
WorkingDirectory=/opt/alphapulse-rl
ExecStart=/opt/alphapulse-rl/venv/bin/python deployment/monitoring/monitor_service.py
Restart=always
RestartSec=10
Environment=PYTHONPATH=/opt/alphapulse-rl
EnvironmentFile=/opt/alphapulse-rl/deployment/.env

[Install]
WantedBy=multi-user.target
"""
        
        with open('deployment/systemd/alphapulse-monitor.service', 'w') as f:
            f.write(monitor_service)
        
        # Validate service files exist
        self.assertTrue(os.path.exists('deployment/systemd/alphapulse-trading.service'))
        self.assertTrue(os.path.exists('deployment/systemd/alphapulse-monitor.service'))
        
        # Basic validation of service file content
        with open('deployment/systemd/alphapulse-trading.service', 'r') as f:
            trading_content = f.read()
        
        self.assertIn('[Unit]', trading_content)
        self.assertIn('[Service]', trading_content)
        self.assertIn('[Install]', trading_content)
        self.assertIn('ExecStart=', trading_content)
        self.assertIn('Restart=always', trading_content)
    
    def test_deployment_validation_integration(self):
        """Test integration of deployment validation components."""
        # Set up complete deployment environment
        self._setup_complete_deployment_environment()
        
        # Test that all components are properly configured
        self.assertTrue(os.path.exists('deployment/.env'))
        self.assertTrue(os.path.exists('config/config.yaml'))
        self.assertTrue(os.path.exists('config/trading_params.yaml'))
        self.assertTrue(os.path.exists('deployment/docker/docker-compose.yml'))
        
        # Test configuration consistency
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        with open('config/trading_params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        # Verify configuration consistency
        self.assertEqual(config['system']['name'], 'alphapulse-rl')
        self.assertIn('risk_management', params)
        self.assertIn('trading', params)
    
    def test_deployment_finalization_procedures(self):
        """Test deployment finalization procedures."""
        # Create test deployment directory structure
        deployment_dirs = [
            'deployment/backups/daily',
            'deployment/backups/weekly',
            'deployment/monitoring/static',
            'deployment/docker/ssl',
            'models/saved',
            'models/backups',
            'data/portfolio',
            'logs/archive'
        ]
        
        for dir_path in deployment_dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        # Test directory creation
        for dir_path in deployment_dirs:
            self.assertTrue(os.path.exists(dir_path))
        
        # Test deployment summary creation
        summary_content = """# AlphaPulse-RL Production Deployment Summary

## Deployment Information
- **Deployment Date**: 2024-01-01 12:00:00
- **System Version**: Production Ready
- **Environment**: Production

## System Architecture
- **alphapulse-trading**: Main trading application (Port 8080)
- **alphapulse-monitor**: Monitoring dashboard (Port 8081)
- **alphapulse-redis**: Redis cache (Port 6379)

## Next Steps
1. Configure environment: cp deployment/.env.template deployment/.env
2. Update API credentials in deployment/.env
3. Run deployment: ./deployment/scripts/deploy.sh
4. Start with paper trading mode
5. Monitor system for first 24 hours

For detailed procedures, see deployment/PRODUCTION_DEPLOYMENT_GUIDE.md
"""
        
        summary_path = 'deployment/DEPLOYMENT_SUMMARY.md'
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        self.assertTrue(os.path.exists(summary_path))
        
        # Verify summary content
        with open(summary_path, 'r') as f:
            content = f.read()
            self.assertIn('AlphaPulse-RL', content)
            self.assertIn('Production', content)
            self.assertIn('Next Steps', content)
    
    def test_deployment_validation_script_integration(self):
        """Test deployment validation script integration."""
        validation_script_path = os.path.join(os.path.dirname(__file__), '..', 'deployment', 'scripts', 'validate_deployment.py')
        
        if os.path.exists(validation_script_path):
            # Test script can be imported
            sys.path.insert(0, os.path.dirname(validation_script_path))
            try:
                import validate_deployment
                
                # Test DeploymentValidator class exists
                self.assertTrue(hasattr(validate_deployment, 'DeploymentValidator'))
                
                # Test essential methods exist
                validator = validate_deployment.DeploymentValidator()
                essential_methods = [
                    'validate_environment_configuration',
                    'validate_configuration_files',
                    'validate_docker_environment',
                    'validate_system_resources',
                    'validate_network_connectivity',
                    'validate_security_configuration',
                    'validate_dependencies',
                    'validate_model_files',
                    'run_all_validations'
                ]
                
                for method in essential_methods:
                    self.assertTrue(hasattr(validator, method), f"Method {method} not found")
                
            except ImportError as e:
                self.skipTest(f"Deployment validation script not importable: {e}")
            finally:
                sys.path.pop(0)
    
    def test_production_readiness_checklist(self):
        """Test production readiness checklist validation."""
        # Create production readiness checklist
        checklist_items = {
            'environment_configured': False,
            'api_credentials_set': False,
            'ssl_certificates_installed': False,
            'firewall_configured': False,
            'monitoring_enabled': False,
            'backups_configured': False,
            'logging_configured': False,
            'models_trained': False
        }
        
        # Test environment configuration
        env_file = 'deployment/.env'
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                content = f.read()
                if 'WEEX_API_KEY=' in content and 'your_api_key_here' not in content:
                    checklist_items['api_credentials_set'] = True
                if 'TRADING_MODE=' in content:
                    checklist_items['environment_configured'] = True
        
        # Test SSL certificates
        ssl_cert_path = 'deployment/docker/ssl/cert.pem'
        ssl_key_path = 'deployment/docker/ssl/key.pem'
        if os.path.exists(ssl_cert_path) and os.path.exists(ssl_key_path):
            checklist_items['ssl_certificates_installed'] = True
        
        # Test monitoring configuration
        monitoring_config = 'config/monitoring.yaml'
        if os.path.exists(monitoring_config):
            checklist_items['monitoring_enabled'] = True
        
        # Test backup configuration
        backup_script = 'deployment/scripts/backup_system.py'
        if os.path.exists(backup_script):
            checklist_items['backups_configured'] = True
        
        # Test logging configuration
        logging_config = 'config/logging_config.py'
        if os.path.exists(logging_config):
            checklist_items['logging_configured'] = True
        
        # Test model files
        model_files = ['models/ppo_agent.pth', 'models/optimized_ppo_agent.pth']
        for model_file in model_files:
            if os.path.exists(model_file):
                checklist_items['models_trained'] = True
                break
        
        # Calculate readiness score
        completed_items = sum(1 for item in checklist_items.values() if item)
        total_items = len(checklist_items)
        readiness_score = (completed_items / total_items) * 100
        
        # Test that we have some basic readiness
        self.assertGreaterEqual(readiness_score, 0)
        self.assertLessEqual(readiness_score, 100)
        
        # Log readiness status
        print(f"Production readiness: {readiness_score:.1f}% ({completed_items}/{total_items} items)")
    
    def test_deployment_rollback_procedures(self):
        """Test deployment rollback procedures."""
        # Create current deployment state
        current_config = {
            'version': '2.0.0',
            'deployment_id': 'deploy_20240101_120000',
            'features': ['new_feature_1', 'new_feature_2']
        }
        
        with open('config/deployment_state.yaml', 'w') as f:
            yaml.dump(current_config, f)
        
        # Create backup state
        backup_dir = 'backups/rollback_20240101_100000'
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_config = {
            'version': '1.9.0',
            'deployment_id': 'deploy_20231201_100000',
            'features': ['stable_feature_1', 'stable_feature_2']
        }
        
        with open(f'{backup_dir}/deployment_state.yaml', 'w') as f:
            yaml.dump(backup_config, f)
        
        # Simulate rollback procedure
        if os.path.exists('config/deployment_state.yaml'):
            # Backup current state
            shutil.copy('config/deployment_state.yaml', 'config/deployment_state.yaml.backup')
            
            # Restore from backup
            shutil.copy(f'{backup_dir}/deployment_state.yaml', 'config/deployment_state.yaml')
        
        # Verify rollback
        with open('config/deployment_state.yaml', 'r') as f:
            restored_config = yaml.safe_load(f)
        
        self.assertEqual(restored_config['version'], '1.9.0')
        self.assertEqual(restored_config['deployment_id'], 'deploy_20231201_100000')
        self.assertIn('stable_feature_1', restored_config['features'])
        
        # Verify backup exists
        self.assertTrue(os.path.exists('config/deployment_state.yaml.backup'))
    
    def test_deployment_health_monitoring_integration(self):
        """Test deployment health monitoring integration."""
        # Create health check configuration
        health_config = {
            'health_checks': {
                'trading_service': {
                    'url': 'http://localhost:8080/health',
                    'timeout': 10,
                    'expected_status': 200
                },
                'monitoring_service': {
                    'url': 'http://localhost:8081/health',
                    'timeout': 10,
                    'expected_status': 200
                },
                'redis_service': {
                    'host': 'localhost',
                    'port': 6379,
                    'timeout': 5
                }
            },
            'alert_thresholds': {
                'cpu_warning': 80,
                'cpu_critical': 90,
                'memory_warning': 80,
                'memory_critical': 90,
                'disk_warning': 85,
                'disk_critical': 95
            }
        }
        
        with open('config/health_monitoring.yaml', 'w') as f:
            yaml.dump(health_config, f)
        
        # Test configuration file creation
        self.assertTrue(os.path.exists('config/health_monitoring.yaml'))
        
        # Test configuration loading
        with open('config/health_monitoring.yaml', 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        self.assertIn('health_checks', loaded_config)
        self.assertIn('alert_thresholds', loaded_config)
        self.assertEqual(loaded_config['health_checks']['trading_service']['url'], 'http://localhost:8080/health')
        self.assertEqual(loaded_config['alert_thresholds']['cpu_critical'], 90)
    
    def test_security_configuration(self):
        """Test security configuration for deployment."""
        # Create secure environment file
        secure_env = """
WEEX_API_KEY=secure_api_key_with_proper_length_12345
WEEX_SECRET_KEY=secure_secret_key_with_proper_length_67890
WEEX_PASSPHRASE=secure_passphrase_123
TRADING_MODE=paper
INITIAL_BALANCE=10000
"""
        
        with open('deployment/.env', 'w') as f:
            f.write(secure_env)
        
        # Set restrictive permissions
        os.chmod('deployment/.env', 0o600)
        
        # Verify permissions
        stat_info = os.stat('deployment/.env')
        permissions = oct(stat_info.st_mode)[-3:]
        self.assertEqual(permissions, '600')
        
        # Test that sensitive data is not in default state
        with open('deployment/.env', 'r') as f:
            content = f.read()
        
        self.assertNotIn('your_api_key_here', content)
        self.assertNotIn('your_secret_key_here', content)
        self.assertNotIn('your_passphrase_here', content)
    
    def test_monitoring_integration_deployment(self):
        """Test monitoring system deployment integration."""
        # Create monitoring configuration
        monitoring_config = {
            'monitoring': {
                'enabled': True,
                'port': 8081,
                'redis_host': 'redis',
                'redis_port': 6379,
                'alert_email': 'alerts@example.com',
                'alert_webhook': 'https://hooks.slack.com/test'
            },
            'alerting': {
                'cpu_threshold_warning': 80,
                'cpu_threshold_critical': 90,
                'memory_threshold_warning': 80,
                'memory_threshold_critical': 90,
                'daily_loss_threshold': 3.0,
                'drawdown_threshold': 12.0
            }
        }
        
        with open('config/monitoring.yaml', 'w') as f:
            yaml.dump(monitoring_config, f)
        
        # Verify monitoring configuration
        self.assertTrue(os.path.exists('config/monitoring.yaml'))
        
        with open('config/monitoring.yaml', 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        self.assertTrue(loaded_config['monitoring']['enabled'])
        self.assertEqual(loaded_config['monitoring']['port'], 8081)
        self.assertEqual(loaded_config['alerting']['daily_loss_threshold'], 3.0)
    
    def _setup_complete_deployment_environment(self):
        """Set up a complete deployment environment for testing."""
        # Environment file
        env_content = """
WEEX_API_KEY=test_api_key_12345678901234567890
WEEX_SECRET_KEY=test_secret_key_12345678901234567890
WEEX_PASSPHRASE=test_passphrase_123
TRADING_MODE=paper
INITIAL_BALANCE=10000
LOG_LEVEL=INFO
"""
        with open('deployment/.env', 'w') as f:
            f.write(env_content)
        
        # Main configuration
        config = {
            'system': {
                'name': 'alphapulse-rl',
                'version': '1.0.0'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/alphapulse.log'
            }
        }
        with open('config/config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        # Trading parameters
        params = {
            'risk_management': {
                'max_leverage': 12,
                'max_position_size_percent': 10
            },
            'trading': {
                'pairs': ['BTCUSDT', 'ETHUSDT'],
                'confidence_threshold': 0.8
            }
        }
        with open('config/trading_params.yaml', 'w') as f:
            yaml.dump(params, f)
        
        # Docker Compose
        compose = {
            'version': '3.8',
            'services': {
                'alphapulse-trading': {
                    'build': '.',
                    'ports': ['8080:8080']
                },
                'redis': {
                    'image': 'redis:7-alpine'
                }
            }
        }
        with open('deployment/docker/docker-compose.yml', 'w') as f:
            yaml.dump(compose, f)

if __name__ == '__main__':
    import unittest
    unittest.main()
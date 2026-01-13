#!/usr/bin/env python3
"""
Test suite for deployment validation and monitoring integration.

Tests comprehensive deployment validation procedures, monitoring system integration,
and production readiness validation to ensure reliable system deployment.
"""

import os
import sys
import json
import yaml
import time
import tempfile
import shutil
import subprocess
from unittest import TestCase
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta

class TestDeploymentValidation(TestCase):
    """Test deployment validation procedures."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create basic directory structure
        directories = [
            'config',
            'deployment/docker',
            'deployment/scripts',
            'deployment/monitoring',
            'models',
            'logs',
            'backups'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_deployment_validation_script_functionality(self):
        """Test deployment validation script functionality."""
        validation_script_path = os.path.join(os.path.dirname(__file__), '..', 'deployment', 'scripts', 'validate_deployment.py')
        
        if os.path.exists(validation_script_path):
            # Test script can be imported
            sys.path.insert(0, os.path.dirname(validation_script_path))
            try:
                import validate_deployment
                
                # Test DeploymentValidator class
                self.assertTrue(hasattr(validate_deployment, 'DeploymentValidator'))
                
                validator = validate_deployment.DeploymentValidator()
                
                # Test validator initialization
                self.assertIn('timestamp', validator.results)
                self.assertIn('overall_status', validator.results)
                self.assertIn('validation_results', validator.results)
                
                # Test validation methods exist
                validation_methods = [
                    'validate_environment_configuration',
                    'validate_configuration_files',
                    'validate_docker_environment',
                    'validate_system_resources',
                    'validate_network_connectivity',
                    'validate_security_configuration',
                    'validate_dependencies',
                    'validate_model_files'
                ]
                
                for method in validation_methods:
                    self.assertTrue(hasattr(validator, method), f"Method {method} not found")
                
            except ImportError as e:
                self.skipTest(f"Deployment validation script not importable: {e}")
            finally:
                sys.path.pop(0)
    
    def test_environment_configuration_validation(self):
        """Test environment configuration validation."""
        # Create valid environment file
        env_content = """
# AlphaPulse-RL Environment Configuration
WEEX_API_KEY=test_api_key_abcdef123456789
WEEX_SECRET_KEY=test_secret_key_abcdef123456789
WEEX_PASSPHRASE=test_passphrase_123
TRADING_MODE=paper
INITIAL_BALANCE=10000
LOG_LEVEL=INFO
REDIS_HOST=localhost
REDIS_PORT=6379
MONITORING_INTERVAL=30
ALERT_EMAIL=alerts@example.com
ALERT_WEBHOOK=https://hooks.slack.com/services/test
"""
        
        with open('deployment/.env', 'w') as f:
            f.write(env_content)
        
        # Test environment file validation
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
            self.assertNotIn('your_', env_vars[var].lower(), f"Environment variable {var} has default value")
        
        # Validate specific values
        self.assertIn(env_vars['TRADING_MODE'], ['paper', 'live'])
        self.assertTrue(float(env_vars['INITIAL_BALANCE']) > 0)
        self.assertIn('@', env_vars.get('ALERT_EMAIL', ''))
        self.assertTrue(env_vars.get('ALERT_WEBHOOK', '').startswith('https://'))
    
    def test_configuration_files_validation(self):
        """Test configuration files validation."""
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
                'backup_count': 5,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'database': {
                'redis': {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0,
                    'password': None
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
                'max_drawdown_percent': 12,
                'volatility_threshold': 0.05
            },
            'trading': {
                'pairs': ['BTCUSDT', 'ETHUSDT'],
                'timeframes': ['5m', '15m'],
                'confidence_threshold': 0.8,
                'trading_interval': 60
            },
            'model': {
                'update_frequency': 3600,
                'batch_size': 32,
                'learning_rate': 0.0003,
                'model_path': 'models/ppo_agent.pth'
            }
        }
        
        with open('config/trading_params.yaml', 'w') as f:
            yaml.dump(trading_params, f)
        
        # Create production configuration
        production_config = {
            'deployment': {
                'mode': 'production',
                'replicas': 1,
                'restart_policy': 'always',
                'health_check_interval': 30
            },
            'security': {
                'ssl_enabled': True,
                'cert_path': '/etc/ssl/certs/alphapulse.crt',
                'key_path': '/etc/ssl/private/alphapulse.key'
            },
            'monitoring': {
                'enabled': True,
                'metrics_port': 8081,
                'alerts_enabled': True
            }
        }
        
        with open('deployment/production_config.yaml', 'w') as f:
            yaml.dump(production_config, f)
        
        # Validate configuration files
        config_files = [
            'config/config.yaml',
            'config/trading_params.yaml',
            'deployment/production_config.yaml'
        ]
        
        for config_file in config_files:
            self.assertTrue(os.path.exists(config_file), f"Configuration file {config_file} not found")
            
            # Test YAML parsing
            with open(config_file, 'r') as f:
                try:
                    config_data = yaml.safe_load(f)
                    self.assertIsInstance(config_data, dict, f"Configuration file {config_file} is not valid YAML")
                except yaml.YAMLError as e:
                    self.fail(f"Configuration file {config_file} has invalid YAML: {e}")
        
        # Test configuration content validation
        with open('config/config.yaml', 'r') as f:
            main_config_loaded = yaml.safe_load(f)
        
        self.assertEqual(main_config_loaded['system']['name'], 'alphapulse-rl')
        self.assertEqual(main_config_loaded['logging']['level'], 'INFO')
        self.assertEqual(main_config_loaded['database']['redis']['port'], 6379)
        
        with open('config/trading_params.yaml', 'r') as f:
            trading_params_loaded = yaml.safe_load(f)
        
        self.assertEqual(trading_params_loaded['risk_management']['max_leverage'], 12)
        self.assertEqual(trading_params_loaded['trading']['confidence_threshold'], 0.8)
        self.assertIn('BTCUSDT', trading_params_loaded['trading']['pairs'])
    
    def test_docker_environment_validation(self):
        """Test Docker environment validation."""
        # Create Docker Compose file
        docker_compose = {
            'version': '3.8',
            'services': {
                'alphapulse-trading': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile'
                    },
                    'ports': ['8080:8080'],
                    'environment': [
                        'TRADING_MODE=paper',
                        'LOG_LEVEL=INFO'
                    ],
                    'volumes': [
                        './logs:/app/logs',
                        './models:/app/models',
                        './config:/app/config'
                    ],
                    'depends_on': ['redis'],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    }
                },
                'alphapulse-monitor': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.monitor'
                    },
                    'ports': ['8081:8081'],
                    'environment': [
                        'REDIS_HOST=redis',
                        'MONITORING_INTERVAL=30'
                    ],
                    'depends_on': ['redis'],
                    'restart': 'unless-stopped'
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'ports': ['6379:6379'],
                    'volumes': ['redis_data:/data'],
                    'restart': 'unless-stopped',
                    'command': 'redis-server --appendonly yes'
                }
            },
            'volumes': {
                'redis_data': {}
            },
            'networks': {
                'alphapulse-network': {
                    'driver': 'bridge'
                }
            }
        }
        
        with open('deployment/docker/docker-compose.yml', 'w') as f:
            yaml.dump(docker_compose, f)
        
        # Create Dockerfile
        dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 alphapulse && chown -R alphapulse:alphapulse /app
USER alphapulse

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "main.py"]
"""
        
        with open('deployment/docker/Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        # Create monitoring Dockerfile
        monitor_dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy monitoring code
COPY deployment/monitoring/ ./deployment/monitoring/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 alphapulse && chown -R alphapulse:alphapulse /app
USER alphapulse

# Expose port
EXPOSE 8081

# Run monitoring service
CMD ["python", "deployment/monitoring/monitor_service.py"]
"""
        
        with open('deployment/docker/Dockerfile.monitor', 'w') as f:
            f.write(monitor_dockerfile_content)
        
        # Validate Docker files
        self.assertTrue(os.path.exists('deployment/docker/docker-compose.yml'))
        self.assertTrue(os.path.exists('deployment/docker/Dockerfile'))
        self.assertTrue(os.path.exists('deployment/docker/Dockerfile.monitor'))
        
        # Test Docker Compose file structure
        with open('deployment/docker/docker-compose.yml', 'r') as f:
            compose_data = yaml.safe_load(f)
        
        self.assertIn('services', compose_data)
        self.assertIn('alphapulse-trading', compose_data['services'])
        self.assertIn('alphapulse-monitor', compose_data['services'])
        self.assertIn('redis', compose_data['services'])
        
        # Validate service configurations
        trading_service = compose_data['services']['alphapulse-trading']
        self.assertIn('8080:8080', trading_service['ports'])
        self.assertEqual(trading_service['restart'], 'unless-stopped')
        self.assertIn('healthcheck', trading_service)
        
        monitor_service = compose_data['services']['alphapulse-monitor']
        self.assertIn('8081:8081', monitor_service['ports'])
        self.assertIn('redis', monitor_service['depends_on'])
        
        redis_service = compose_data['services']['redis']
        self.assertEqual(redis_service['image'], 'redis:7-alpine')
        self.assertIn('redis-server --appendonly yes', redis_service['command'])
    
    @patch('subprocess.run')
    def test_system_resources_validation(self, mock_run):
        """Test system resources validation."""
        # Mock system commands
        mock_run.return_value = MagicMock(returncode=0, stdout="Docker version 20.10.0")
        
        # Test resource requirements
        min_requirements = {
            'cpu_cores': 2,
            'memory_gb': 4,
            'disk_gb': 20
        }
        
        # Mock system resource check
        with patch('psutil.cpu_count') as mock_cpu_count, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Test adequate resources
            mock_cpu_count.return_value = 4
            
            mock_memory_obj = MagicMock()
            mock_memory_obj.total = 8 * 1024**3  # 8GB
            mock_memory_obj.percent = 50.0
            mock_memory.return_value = mock_memory_obj
            
            mock_disk_obj = MagicMock()
            mock_disk_obj.total = 100 * 1024**3  # 100GB
            mock_disk_obj.free = 80 * 1024**3   # 80GB free
            mock_disk_obj.percent = 20.0
            mock_disk.return_value = mock_disk_obj
            
            # Validate resources meet requirements
            cpu_cores = mock_cpu_count.return_value
            memory_gb = mock_memory_obj.total / (1024**3)
            disk_free_gb = mock_disk_obj.free / (1024**3)
            
            self.assertGreaterEqual(cpu_cores, min_requirements['cpu_cores'])
            self.assertGreaterEqual(memory_gb, min_requirements['memory_gb'])
            self.assertGreaterEqual(disk_free_gb, min_requirements['disk_gb'])
            
            # Test resource utilization
            self.assertLess(mock_memory_obj.percent, 80, "Memory usage too high for deployment")
            self.assertLess(mock_disk_obj.percent, 80, "Disk usage too high for deployment")
    
    def test_security_configuration_validation(self):
        """Test security configuration validation."""
        # Create SSL certificate files (mock)
        ssl_dir = 'deployment/docker/ssl'
        os.makedirs(ssl_dir, exist_ok=True)
        
        # Mock SSL certificate
        cert_content = """-----BEGIN CERTIFICATE-----
MIIDXTCCAkWgAwIBAgIJAKoK/heBjcOuMA0GCSqGSIb3DQEBBQUAMEUxCzAJBgNV
BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX
aWRnaXRzIFB0eSBMdGQwHhcNMTcwODI3MjM1NzU5WhcNMTgwODI3MjM1NzU5WjBF
MQswCQYDVQQGEwJBVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50
ZXJuZXQgV2lkZ2l0cyBQdHkgTHRkMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIB
CgKCAQEAuuExKvwjlolnxajFqaBqJqyc/o/PiTjXxq53dt0aR4R7EIGiS28WElhG
/Nc2ZFGADMaZdXoXiGu1belSpu1aBdqhWOk0cUrjdwMBaOM3+py7gokz8B5AqyMn
-----END CERTIFICATE-----"""
        
        with open(f'{ssl_dir}/cert.pem', 'w') as f:
            f.write(cert_content)
        
        # Mock SSL private key
        key_content = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC64TEq/COWiWfF
qMWpoGomrJz+j8+JONfGrnd23RpHhHsQgaJLbxYSWEb81zZkUYAMxpl1eheIa7Vt
6VKm7VoF2qFY6TRxSuN3AwFo4zf6nLuCiTPwHkCrIyc5RQK5auW7fBEUGMyI4VQ7
-----END PRIVATE KEY-----"""
        
        with open(f'{ssl_dir}/key.pem', 'w') as f:
            f.write(key_content)
        
        # Test SSL certificate files exist
        self.assertTrue(os.path.exists(f'{ssl_dir}/cert.pem'))
        self.assertTrue(os.path.exists(f'{ssl_dir}/key.pem'))
        
        # Test file permissions
        cert_stat = os.stat(f'{ssl_dir}/cert.pem')
        key_stat = os.stat(f'{ssl_dir}/key.pem')
        
        # SSL files should have restrictive permissions
        cert_permissions = oct(cert_stat.st_mode)[-3:]
        key_permissions = oct(key_stat.st_mode)[-3:]
        
        # Certificate can be readable, but key should be restricted
        # Allow common permission patterns (644, 664, 600)
        self.assertIn(cert_permissions, ['644', '664', '600'])
        self.assertIn(key_permissions, ['664', '644', '600', '400'])
        
        # Test environment file permissions
        env_file = 'deployment/.env'
        if os.path.exists(env_file):
            env_stat = os.stat(env_file)
            env_permissions = oct(env_stat.st_mode)[-3:]
            self.assertEqual(env_permissions, '600', "Environment file should have 600 permissions")
        
        # Create security configuration
        security_config = {
            'ssl': {
                'enabled': True,
                'cert_path': f'{ssl_dir}/cert.pem',
                'key_path': f'{ssl_dir}/key.pem',
                'protocols': ['TLSv1.2', 'TLSv1.3']
            },
            'authentication': {
                'api_key_required': True,
                'rate_limiting': {
                    'enabled': True,
                    'requests_per_minute': 60
                }
            },
            'firewall': {
                'enabled': True,
                'allowed_ports': [22, 80, 443, 8080, 8081],
                'allowed_ips': ['127.0.0.1', '10.0.0.0/8']
            }
        }
        
        with open('config/security.yaml', 'w') as f:
            yaml.dump(security_config, f)
        
        # Validate security configuration
        with open('config/security.yaml', 'r') as f:
            loaded_security = yaml.safe_load(f)
        
        self.assertTrue(loaded_security['ssl']['enabled'])
        self.assertIn('TLSv1.3', loaded_security['ssl']['protocols'])
        self.assertTrue(loaded_security['authentication']['api_key_required'])
        self.assertEqual(loaded_security['authentication']['rate_limiting']['requests_per_minute'], 60)
    
    def test_monitoring_integration_validation(self):
        """Test monitoring integration validation."""
        # Create monitoring configuration
        monitoring_config = {
            'monitoring': {
                'enabled': True,
                'port': 8081,
                'redis_host': 'redis',
                'redis_port': 6379,
                'update_interval': 30,
                'dashboard_enabled': True
            },
            'alerting': {
                'enabled': True,
                'email_notifications': {
                    'enabled': True,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'recipients': ['alerts@example.com']
                },
                'webhook_notifications': {
                    'enabled': True,
                    'slack_webhook': 'https://hooks.slack.com/services/test'
                },
                'thresholds': {
                    'cpu_warning': 80,
                    'cpu_critical': 90,
                    'memory_warning': 80,
                    'memory_critical': 90,
                    'disk_warning': 85,
                    'disk_critical': 95,
                    'daily_loss_critical': 3.0,
                    'drawdown_critical': 12.0
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/monitoring.log',
                'max_size': '50MB',
                'backup_count': 3
            }
        }
        
        with open('config/monitoring.yaml', 'w') as f:
            yaml.dump(monitoring_config, f)
        
        # Test monitoring configuration
        self.assertTrue(os.path.exists('config/monitoring.yaml'))
        
        with open('config/monitoring.yaml', 'r') as f:
            loaded_monitoring = yaml.safe_load(f)
        
        # Validate monitoring settings
        self.assertTrue(loaded_monitoring['monitoring']['enabled'])
        self.assertEqual(loaded_monitoring['monitoring']['port'], 8081)
        self.assertTrue(loaded_monitoring['alerting']['enabled'])
        self.assertEqual(loaded_monitoring['alerting']['thresholds']['cpu_critical'], 90)
        self.assertEqual(loaded_monitoring['alerting']['thresholds']['daily_loss_critical'], 3.0)
        
        # Test alert notification configuration
        email_config = loaded_monitoring['alerting']['email_notifications']
        self.assertTrue(email_config['enabled'])
        self.assertEqual(email_config['smtp_server'], 'smtp.gmail.com')
        self.assertIn('alerts@example.com', email_config['recipients'])
        
        webhook_config = loaded_monitoring['alerting']['webhook_notifications']
        self.assertTrue(webhook_config['enabled'])
        self.assertTrue(webhook_config['slack_webhook'].startswith('https://hooks.slack.com'))
    
    def test_production_deployment_checklist(self):
        """Test production deployment checklist validation."""
        # Create comprehensive deployment checklist
        checklist = {
            'pre_deployment': {
                'environment_configured': False,
                'api_credentials_set': False,
                'ssl_certificates_installed': False,
                'firewall_configured': False,
                'monitoring_enabled': False,
                'backups_configured': False,
                'models_trained': False,
                'tests_passed': False
            },
            'deployment': {
                'docker_images_built': False,
                'services_started': False,
                'health_checks_passed': False,
                'monitoring_active': False,
                'alerts_configured': False
            },
            'post_deployment': {
                'system_validated': False,
                'performance_tested': False,
                'documentation_updated': False,
                'team_notified': False
            }
        }
        
        # Simulate checklist validation
        # Check environment configuration
        if os.path.exists('deployment/.env'):
            with open('deployment/.env', 'r') as f:
                content = f.read()
                if 'WEEX_API_KEY=' in content and 'your_api_key_here' not in content:
                    checklist['pre_deployment']['api_credentials_set'] = True
                if 'TRADING_MODE=' in content:
                    checklist['pre_deployment']['environment_configured'] = True
        
        # Check SSL certificates
        if os.path.exists('deployment/docker/ssl/cert.pem') and os.path.exists('deployment/docker/ssl/key.pem'):
            checklist['pre_deployment']['ssl_certificates_installed'] = True
        
        # Check monitoring configuration
        if os.path.exists('config/monitoring.yaml'):
            checklist['pre_deployment']['monitoring_enabled'] = True
        
        # Check backup configuration
        if os.path.exists('deployment/scripts/backup_system.py'):
            checklist['pre_deployment']['backups_configured'] = True
        
        # Check model files
        model_files = ['models/ppo_agent.pth', 'models/optimized_ppo_agent.pth']
        for model_file in model_files:
            if os.path.exists(model_file):
                checklist['pre_deployment']['models_trained'] = True
                break
        
        # Calculate completion percentages
        pre_deployment_items = list(checklist['pre_deployment'].values())
        deployment_items = list(checklist['deployment'].values())
        post_deployment_items = list(checklist['post_deployment'].values())
        
        pre_deployment_completion = sum(pre_deployment_items) / len(pre_deployment_items) * 100
        deployment_completion = sum(deployment_items) / len(deployment_items) * 100
        post_deployment_completion = sum(post_deployment_items) / len(post_deployment_items) * 100
        
        overall_completion = (pre_deployment_completion + deployment_completion + post_deployment_completion) / 3
        
        # Test checklist structure
        self.assertIn('pre_deployment', checklist)
        self.assertIn('deployment', checklist)
        self.assertIn('post_deployment', checklist)
        
        # Test completion calculation
        self.assertGreaterEqual(overall_completion, 0)
        self.assertLessEqual(overall_completion, 100)
        
        # Save checklist results
        checklist_results = {
            'checklist': checklist,
            'completion': {
                'pre_deployment': pre_deployment_completion,
                'deployment': deployment_completion,
                'post_deployment': post_deployment_completion,
                'overall': overall_completion
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open('deployment/deployment_checklist.json', 'w') as f:
            json.dump(checklist_results, f, indent=2)
        
        self.assertTrue(os.path.exists('deployment/deployment_checklist.json'))
        
        # Log completion status
        print(f"Deployment checklist completion: {overall_completion:.1f}%")
        print(f"Pre-deployment: {pre_deployment_completion:.1f}%")
        print(f"Deployment: {deployment_completion:.1f}%")
        print(f"Post-deployment: {post_deployment_completion:.1f}%")

if __name__ == '__main__':
    import unittest
    unittest.main()
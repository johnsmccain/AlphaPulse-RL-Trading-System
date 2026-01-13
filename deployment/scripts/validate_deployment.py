#!/usr/bin/env python3
"""
AlphaPulse-RL Deployment Validation Script

Comprehensive validation of production deployment including configuration,
services, security, and operational readiness.
"""

import os
import sys
import json
import yaml
import time
import requests
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Validates production deployment readiness."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'validation_results': {},
            'recommendations': [],
            'critical_issues': [],
            'warnings': []
        }
        
        self.project_root = os.getcwd()
    
    def validate_environment_configuration(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate environment configuration."""
        logger.info("Validating environment configuration...")
        
        details = {}
        issues = []
        
        # Check .env file exists
        env_file = 'deployment/.env'
        if not os.path.exists(env_file):
            issues.append("Environment file not found: deployment/.env")
            return False, "Environment file missing", details
        
        # Read and validate environment variables
        required_vars = [
            'WEEX_API_KEY',
            'WEEX_SECRET_KEY', 
            'WEEX_PASSPHRASE',
            'TRADING_MODE',
            'INITIAL_BALANCE'
        ]
        
        env_vars = {}
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
        
        missing_vars = []
        default_vars = []
        
        for var in required_vars:
            if var not in env_vars:
                missing_vars.append(var)
            elif 'your_' in env_vars[var] or env_vars[var] == '':
                default_vars.append(var)
        
        if missing_vars:
            issues.append(f"Missing environment variables: {', '.join(missing_vars)}")
        
        if default_vars:
            issues.append(f"Environment variables with default values: {', '.join(default_vars)}")
        
        # Validate trading mode
        trading_mode = env_vars.get('TRADING_MODE', '').lower()
        if trading_mode not in ['paper', 'live']:
            issues.append(f"Invalid trading mode: {trading_mode}. Must be 'paper' or 'live'")
        elif trading_mode == 'live':
            self.results['warnings'].append("Trading mode is set to 'live'. Ensure this is intentional for production.")
        
        details['env_vars_count'] = len(env_vars)
        details['trading_mode'] = trading_mode
        details['missing_vars'] = missing_vars
        details['default_vars'] = default_vars
        
        if issues:
            return False, "; ".join(issues), details
        
        return True, "Environment configuration is valid", details
    
    def validate_configuration_files(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate configuration files."""
        logger.info("Validating configuration files...")
        
        details = {}
        issues = []
        
        config_files = [
            ('config/config.yaml', 'Main configuration'),
            ('config/trading_params.yaml', 'Trading parameters'),
            ('deployment/docker/docker-compose.yml', 'Docker Compose'),
            ('deployment/production_config.yaml', 'Production configuration')
        ]
        
        valid_files = 0
        
        for file_path, description in config_files:
            if os.path.exists(file_path):
                try:
                    if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                        with open(file_path, 'r') as f:
                            yaml.safe_load(f)
                    valid_files += 1
                    details[f"{description.lower().replace(' ', '_')}_valid"] = True
                except Exception as e:
                    issues.append(f"Invalid {description} file ({file_path}): {str(e)}")
                    details[f"{description.lower().replace(' ', '_')}_valid"] = False
            else:
                issues.append(f"Missing {description} file: {file_path}")
                details[f"{description.lower().replace(' ', '_')}_valid"] = False
        
        details['valid_files_count'] = valid_files
        details['total_files_count'] = len(config_files)
        
        if issues:
            return False, "; ".join(issues), details
        
        return True, f"All {len(config_files)} configuration files are valid", details
    
    def validate_docker_environment(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate Docker environment."""
        logger.info("Validating Docker environment...")
        
        details = {}
        issues = []
        
        # Check Docker installation
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                details['docker_version'] = result.stdout.strip()
            else:
                issues.append("Docker is not properly installed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            issues.append("Docker command not found")
        
        # Check Docker Compose installation
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                details['docker_compose_version'] = result.stdout.strip()
            else:
                issues.append("Docker Compose is not properly installed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            issues.append("Docker Compose command not found")
        
        # Check Docker daemon
        try:
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                details['docker_daemon_running'] = True
            else:
                issues.append("Docker daemon is not running")
                details['docker_daemon_running'] = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            issues.append("Cannot check Docker daemon status")
            details['docker_daemon_running'] = False
        
        # Check Docker Compose file
        compose_file = 'deployment/docker/docker-compose.yml'
        if os.path.exists(compose_file):
            try:
                result = subprocess.run(['docker-compose', '-f', compose_file, 'config'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    details['compose_file_valid'] = True
                else:
                    issues.append(f"Docker Compose file validation failed: {result.stderr}")
                    details['compose_file_valid'] = False
            except subprocess.TimeoutExpired:
                issues.append("Docker Compose validation timed out")
                details['compose_file_valid'] = False
        else:
            issues.append("Docker Compose file not found")
            details['compose_file_valid'] = False
        
        if issues:
            return False, "; ".join(issues), details
        
        return True, "Docker environment is properly configured", details
    
    def validate_system_resources(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate system resources."""
        logger.info("Validating system resources...")
        
        details = {}
        issues = []
        warnings = []
        
        try:
            import psutil
            
            # CPU check
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            details['cpu_count'] = cpu_count
            details['cpu_usage_percent'] = cpu_percent
            
            if cpu_count < 2:
                issues.append(f"Insufficient CPU cores: {cpu_count} (minimum 2 recommended)")
            elif cpu_count < 4:
                warnings.append(f"Low CPU cores: {cpu_count} (4+ recommended for production)")
            
            if cpu_percent > 80:
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            details['memory_total_gb'] = memory_gb
            details['memory_usage_percent'] = memory.percent
            
            if memory_gb < 4:
                issues.append(f"Insufficient memory: {memory_gb:.1f}GB (minimum 4GB required)")
            elif memory_gb < 8:
                warnings.append(f"Low memory: {memory_gb:.1f}GB (8GB+ recommended for production)")
            
            if memory.percent > 80:
                warnings.append(f"High memory usage: {memory.percent:.1f}%")
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_gb = disk.total / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            
            details['disk_total_gb'] = disk_gb
            details['disk_free_gb'] = disk_free_gb
            details['disk_usage_percent'] = (disk.used / disk.total) * 100
            
            if disk_free_gb < 10:
                issues.append(f"Insufficient disk space: {disk_free_gb:.1f}GB free")
            elif disk_free_gb < 50:
                warnings.append(f"Low disk space: {disk_free_gb:.1f}GB free")
            
        except ImportError:
            issues.append("psutil not available for system resource checking")
        except Exception as e:
            issues.append(f"System resource check failed: {str(e)}")
        
        # Add warnings to global warnings
        self.results['warnings'].extend(warnings)
        
        if issues:
            return False, "; ".join(issues), details
        
        return True, "System resources are adequate", details
    
    def validate_network_connectivity(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate network connectivity."""
        logger.info("Validating network connectivity...")
        
        details = {}
        issues = []
        
        # Test internet connectivity
        try:
            response = requests.get('https://httpbin.org/ip', timeout=10)
            if response.status_code == 200:
                details['internet_connectivity'] = True
                details['public_ip'] = response.json().get('origin', 'unknown')
            else:
                issues.append("Internet connectivity test failed")
                details['internet_connectivity'] = False
        except Exception as e:
            issues.append(f"Internet connectivity test failed: {str(e)}")
            details['internet_connectivity'] = False
        
        # Test WEEX API connectivity
        try:
            response = requests.get('https://api.weex.com/api/v1/time', timeout=10)
            if response.status_code == 200:
                details['weex_api_connectivity'] = True
                api_time = response.json().get('data', {}).get('time', 0)
                local_time = int(time.time() * 1000)
                time_diff = abs(api_time - local_time)
                details['time_sync_diff_ms'] = time_diff
                
                if time_diff > 5000:  # 5 seconds
                    self.results['warnings'].append(f"System time may be out of sync with WEEX API: {time_diff}ms difference")
            else:
                issues.append("WEEX API connectivity test failed")
                details['weex_api_connectivity'] = False
        except Exception as e:
            issues.append(f"WEEX API connectivity test failed: {str(e)}")
            details['weex_api_connectivity'] = False
        
        # Test local ports
        local_ports = [8080, 8081, 6379]  # Trading, monitoring, Redis
        open_ports = []
        
        for port in local_ports:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    open_ports.append(port)
            except Exception:
                pass
        
        details['open_local_ports'] = open_ports
        
        if issues:
            return False, "; ".join(issues), details
        
        return True, "Network connectivity is working", details
    
    def validate_security_configuration(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate security configuration."""
        logger.info("Validating security configuration...")
        
        details = {}
        issues = []
        warnings = []
        
        # Check file permissions
        sensitive_files = [
            'deployment/.env',
            'config/config.yaml',
            'config/trading_params.yaml'
        ]
        
        for file_path in sensitive_files:
            if os.path.exists(file_path):
                stat_info = os.stat(file_path)
                permissions = oct(stat_info.st_mode)[-3:]
                details[f"{file_path.replace('/', '_')}_permissions"] = permissions
                
                # Check if file is world-readable
                if int(permissions[2]) >= 4:
                    warnings.append(f"File {file_path} is world-readable (permissions: {permissions})")
        
        # Check for default passwords/keys
        env_file = 'deployment/.env'
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                content = f.read()
                
                if 'your_api_key_here' in content:
                    issues.append("Default API key found in environment file")
                
                if 'your_secret_key_here' in content:
                    issues.append("Default secret key found in environment file")
                
                if 'generate_secure_token_here' in content:
                    warnings.append("Default authentication token found in environment file")
        
        # Check SSL certificates
        ssl_cert_path = 'deployment/docker/ssl/cert.pem'
        ssl_key_path = 'deployment/docker/ssl/key.pem'
        
        if os.path.exists(ssl_cert_path) and os.path.exists(ssl_key_path):
            details['ssl_certificates_present'] = True
        else:
            details['ssl_certificates_present'] = False
            warnings.append("SSL certificates not found - HTTPS will not be available")
        
        # Check firewall status (Linux only)
        try:
            result = subprocess.run(['ufw', 'status'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                if 'Status: active' in result.stdout:
                    details['firewall_active'] = True
                else:
                    details['firewall_active'] = False
                    warnings.append("UFW firewall is not active")
            else:
                details['firewall_active'] = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            details['firewall_available'] = False
        
        # Add warnings to global warnings
        self.results['warnings'].extend(warnings)
        
        if issues:
            return False, "; ".join(issues), details
        
        return True, "Security configuration is acceptable", details
    
    def validate_dependencies(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate Python dependencies."""
        logger.info("Validating Python dependencies...")
        
        details = {}
        issues = []
        
        # Check requirements.txt
        requirements_file = 'requirements.txt'
        if not os.path.exists(requirements_file):
            issues.append("requirements.txt file not found")
            return False, "Requirements file missing", details
        
        # Check critical dependencies
        critical_deps = [
            'torch',
            'numpy',
            'pandas',
            'requests',
            'pyyaml',
            'redis',
            'flask',
            'psutil'
        ]
        
        missing_deps = []
        installed_deps = []
        
        for dep in critical_deps:
            try:
                __import__(dep)
                installed_deps.append(dep)
            except ImportError:
                missing_deps.append(dep)
        
        details['installed_dependencies'] = installed_deps
        details['missing_dependencies'] = missing_deps
        details['total_critical_deps'] = len(critical_deps)
        
        if missing_deps:
            issues.append(f"Missing critical dependencies: {', '.join(missing_deps)}")
        
        # Check Python version
        python_version = sys.version_info
        details['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version < (3, 9):
            issues.append(f"Python version too old: {details['python_version']} (3.9+ required)")
        
        if issues:
            return False, "; ".join(issues), details
        
        return True, f"All {len(installed_deps)} critical dependencies are installed", details
    
    def validate_model_files(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate model files."""
        logger.info("Validating model files...")
        
        details = {}
        issues = []
        warnings = []
        
        # Check model directory
        model_dir = 'models'
        if not os.path.exists(model_dir):
            issues.append("Models directory not found")
            return False, "Models directory missing", details
        
        # Look for model files
        model_extensions = ['.pth', '.pkl', '.joblib', '.h5', '.pb']
        model_files = []
        
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if any(file.endswith(ext) for ext in model_extensions):
                    model_files.append(os.path.join(root, file))
        
        details['model_files_found'] = len(model_files)
        details['model_files'] = model_files
        
        if not model_files:
            warnings.append("No trained model files found - system will need training")
        
        # Check for specific expected models
        expected_models = ['ppo_agent.pth', 'optimized_ppo_agent.pth']
        found_models = []
        
        for expected in expected_models:
            model_path = os.path.join(model_dir, expected)
            if os.path.exists(model_path):
                found_models.append(expected)
                # Check file size
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                details[f"{expected}_size_mb"] = size_mb
                
                if size_mb < 0.1:  # Less than 100KB
                    warnings.append(f"Model file {expected} seems too small: {size_mb:.2f}MB")
        
        details['expected_models_found'] = found_models
        
        # Add warnings to global warnings
        self.results['warnings'].extend(warnings)
        
        if issues:
            return False, "; ".join(issues), details
        
        return True, f"Model validation completed - {len(model_files)} model files found", details
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation checks."""
        logger.info("Starting comprehensive deployment validation...")
        
        validations = [
            ('environment_configuration', self.validate_environment_configuration),
            ('configuration_files', self.validate_configuration_files),
            ('docker_environment', self.validate_docker_environment),
            ('system_resources', self.validate_system_resources),
            ('network_connectivity', self.validate_network_connectivity),
            ('security_configuration', self.validate_security_configuration),
            ('dependencies', self.validate_dependencies),
            ('model_files', self.validate_model_files)
        ]
        
        passed = 0
        failed = 0
        
        for validation_name, validation_func in validations:
            try:
                success, message, details = validation_func()
                
                self.results['validation_results'][validation_name] = {
                    'status': 'PASS' if success else 'FAIL',
                    'message': message,
                    'details': details,
                    'timestamp': datetime.now().isoformat()
                }
                
                if success:
                    passed += 1
                else:
                    failed += 1
                    self.results['critical_issues'].append(f"{validation_name}: {message}")
                    
            except Exception as e:
                self.results['validation_results'][validation_name] = {
                    'status': 'ERROR',
                    'message': f"Validation failed with exception: {str(e)}",
                    'details': {},
                    'timestamp': datetime.now().isoformat()
                }
                failed += 1
                self.results['critical_issues'].append(f"{validation_name}: Validation error - {str(e)}")
        
        # Determine overall status
        if failed == 0:
            self.results['overall_status'] = 'READY'
        elif passed > failed:
            self.results['overall_status'] = 'READY_WITH_WARNINGS'
        else:
            self.results['overall_status'] = 'NOT_READY'
        
        # Generate recommendations
        self._generate_recommendations()
        
        self.results['summary'] = {
            'total_validations': len(validations),
            'passed': passed,
            'failed': failed,
            'warnings_count': len(self.results['warnings']),
            'critical_issues_count': len(self.results['critical_issues'])
        }
        
        return self.results
    
    def _generate_recommendations(self):
        """Generate deployment recommendations based on validation results."""
        recommendations = []
        
        # Environment recommendations
        env_result = self.results['validation_results'].get('environment_configuration', {})
        if env_result.get('status') == 'FAIL':
            recommendations.append("Configure all required environment variables in deployment/.env")
        
        # Resource recommendations
        resource_result = self.results['validation_results'].get('system_resources', {})
        if resource_result.get('status') == 'PASS':
            details = resource_result.get('details', {})
            if details.get('memory_total_gb', 0) < 8:
                recommendations.append("Consider upgrading to 8GB+ RAM for better performance")
            if details.get('cpu_count', 0) < 4:
                recommendations.append("Consider upgrading to 4+ CPU cores for better performance")
        
        # Security recommendations
        security_result = self.results['validation_results'].get('security_configuration', {})
        if security_result.get('details', {}).get('ssl_certificates_present') == False:
            recommendations.append("Install SSL certificates for HTTPS access")
        if security_result.get('details', {}).get('firewall_active') == False:
            recommendations.append("Enable and configure firewall for security")
        
        # Model recommendations
        model_result = self.results['validation_results'].get('model_files', {})
        if model_result.get('details', {}).get('model_files_found', 0) == 0:
            recommendations.append("Train and save model files before deployment")
        
        # General recommendations
        if self.results['overall_status'] == 'READY_WITH_WARNINGS':
            recommendations.append("Address warnings before production deployment")
        
        recommendations.append("Start with paper trading mode to validate system behavior")
        recommendations.append("Monitor system closely for the first 24 hours after deployment")
        recommendations.append("Set up automated backups and test restore procedures")
        
        self.results['recommendations'] = recommendations
    
    def print_results(self):
        """Print validation results in a readable format."""
        print(f"\n{'='*80}")
        print(f"AlphaPulse-RL Deployment Validation Report")
        print(f"{'='*80}")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"Summary: {self.results['summary']['passed']}/{self.results['summary']['total_validations']} validations passed")
        print(f"{'='*80}")
        
        # Print validation results
        for validation_name, result in self.results['validation_results'].items():
            status_symbol = "✓" if result['status'] == 'PASS' else "✗" if result['status'] == 'FAIL' else "⚠"
            print(f"{status_symbol} {validation_name.replace('_', ' ').title()}: {result['status']}")
            print(f"   {result['message']}")
            
            # Print key details
            if result.get('details'):
                key_details = []
                details = result['details']
                
                if 'cpu_count' in details:
                    key_details.append(f"CPU: {details['cpu_count']} cores")
                if 'memory_total_gb' in details:
                    key_details.append(f"Memory: {details['memory_total_gb']:.1f}GB")
                if 'docker_version' in details:
                    key_details.append(f"Docker: {details['docker_version'].split()[2]}")
                if 'python_version' in details:
                    key_details.append(f"Python: {details['python_version']}")
                
                if key_details:
                    print(f"   Details: {', '.join(key_details)}")
            
            print()
        
        # Print warnings
        if self.results['warnings']:
            print(f"⚠️  Warnings ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"   • {warning}")
            print()
        
        # Print critical issues
        if self.results['critical_issues']:
            print(f"❌ Critical Issues ({len(self.results['critical_issues'])}):")
            for issue in self.results['critical_issues']:
                print(f"   • {issue}")
            print()
        
        # Print recommendations
        if self.results['recommendations']:
            print(f"💡 Recommendations:")
            for i, recommendation in enumerate(self.results['recommendations'], 1):
                print(f"   {i}. {recommendation}")
            print()
        
        # Print final status
        if self.results['overall_status'] == 'READY':
            print("✅ System is ready for deployment!")
        elif self.results['overall_status'] == 'READY_WITH_WARNINGS':
            print("⚠️  System is ready for deployment but has warnings that should be addressed.")
        else:
            print("❌ System is NOT ready for deployment. Please address critical issues first.")
        
        print(f"{'='*80}\n")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaPulse-RL Deployment Validator')
    parser.add_argument('--output', '-o', help='Output file for detailed results (JSON)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    validator = DeploymentValidator()
    results = validator.run_all_validations()
    
    # Print results unless quiet mode
    if not args.quiet:
        validator.print_results()
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {args.output}")
    
    # Exit with appropriate code
    if results['overall_status'] == 'READY':
        sys.exit(0)
    elif results['overall_status'] == 'READY_WITH_WARNINGS':
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == '__main__':
    main()
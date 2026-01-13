#!/usr/bin/env python3
"""
AlphaPulse-RL System Health Check Script

Comprehensive health check for all system components with detailed reporting.
"""

import os
import sys
import json
import time
import requests
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Tuple
import psutil

class HealthChecker:
    """Comprehensive system health checker."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'checks': {}
        }
    
    def check_docker_services(self) -> Tuple[bool, str]:
        """Check Docker services status."""
        try:
            # Check if Docker is running
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False, "Docker daemon is not running"
            
            # Check Docker Compose services
            compose_file = "deployment/docker/docker-compose.yml"
            if not os.path.exists(compose_file):
                return False, f"Docker Compose file not found: {compose_file}"
            
            result = subprocess.run(['docker-compose', '-f', compose_file, 'ps'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return False, f"Failed to check Docker Compose services: {result.stderr}"
            
            # Parse output to check service status
            lines = result.stdout.strip().split('\n')[2:]  # Skip header lines
            services_down = []
            
            for line in lines:
                if line.strip() and 'Up' not in line:
                    service_name = line.split()[0]
                    services_down.append(service_name)
            
            if services_down:
                return False, f"Services down: {', '.join(services_down)}"
            
            return True, "All Docker services are running"
            
        except subprocess.TimeoutExpired:
            return False, "Docker command timed out"
        except Exception as e:
            return False, f"Docker check failed: {str(e)}"
    
    def check_system_resources(self) -> Tuple[bool, str]:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            issues = []
            
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            if issues:
                return False, "; ".join(issues)
            
            return True, f"Resources OK - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, Disk: {disk.percent:.1f}%"
            
        except Exception as e:
            return False, f"Resource check failed: {str(e)}"
    
    def check_trading_service(self) -> Tuple[bool, str]:
        """Check trading service health."""
        try:
            response = requests.get('http://localhost:8080/health', timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'healthy':
                    return True, "Trading service is healthy"
                else:
                    return False, f"Trading service reports unhealthy: {health_data}"
            else:
                return False, f"Trading service returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to trading service"
        except requests.exceptions.Timeout:
            return False, "Trading service health check timed out"
        except Exception as e:
            return False, f"Trading service check failed: {str(e)}"
    
    def check_monitoring_service(self) -> Tuple[bool, str]:
        """Check monitoring service health."""
        try:
            response = requests.get('http://localhost:8081/health', timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'healthy':
                    return True, "Monitoring service is healthy"
                else:
                    return False, f"Monitoring service reports unhealthy: {health_data}"
            else:
                return False, f"Monitoring service returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to monitoring service"
        except requests.exceptions.Timeout:
            return False, "Monitoring service health check timed out"
        except Exception as e:
            return False, f"Monitoring service check failed: {str(e)}"
    
    def check_redis_service(self) -> Tuple[bool, str]:
        """Check Redis service health."""
        try:
            import redis
            
            client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            response = client.ping()
            
            if response:
                # Check memory usage
                info = client.info('memory')
                used_memory_mb = info['used_memory'] / (1024 * 1024)
                
                return True, f"Redis is healthy - Memory used: {used_memory_mb:.1f}MB"
            else:
                return False, "Redis ping failed"
                
        except ImportError:
            return False, "Redis client not available"
        except Exception as e:
            return False, f"Redis check failed: {str(e)}"
    
    def check_api_connectivity(self) -> Tuple[bool, str]:
        """Check WEEX API connectivity."""
        try:
            # Test basic connectivity to WEEX API
            response = requests.get('https://api.weex.com/api/v1/time', timeout=10)
            
            if response.status_code == 200:
                return True, "WEEX API is accessible"
            else:
                return False, f"WEEX API returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to WEEX API"
        except requests.exceptions.Timeout:
            return False, "WEEX API connection timed out"
        except Exception as e:
            return False, f"API connectivity check failed: {str(e)}"
    
    def check_log_files(self) -> Tuple[bool, str]:
        """Check log files and disk usage."""
        try:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                return False, f"Log directory not found: {log_dir}"
            
            # Check log file sizes
            large_files = []
            total_size = 0
            
            for filename in os.listdir(log_dir):
                filepath = os.path.join(log_dir, filename)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    total_size += size
                    
                    # Flag files larger than 100MB
                    if size > 100 * 1024 * 1024:
                        large_files.append(f"{filename} ({size / (1024*1024):.1f}MB)")
            
            issues = []
            
            if large_files:
                issues.append(f"Large log files: {', '.join(large_files)}")
            
            # Check if total log size is over 1GB
            if total_size > 1024 * 1024 * 1024:
                issues.append(f"Total log size: {total_size / (1024*1024*1024):.1f}GB")
            
            if issues:
                return False, "; ".join(issues)
            
            return True, f"Log files OK - Total size: {total_size / (1024*1024):.1f}MB"
            
        except Exception as e:
            return False, f"Log file check failed: {str(e)}"
    
    def check_configuration(self) -> Tuple[bool, str]:
        """Check configuration files."""
        try:
            required_files = [
                'config/config.yaml',
                'config/trading_params.yaml',
                'deployment/.env'
            ]
            
            missing_files = []
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                return False, f"Missing configuration files: {', '.join(missing_files)}"
            
            # Check environment file for required variables
            env_file = 'deployment/.env'
            required_vars = ['WEEX_API_KEY', 'WEEX_SECRET_KEY', 'WEEX_PASSPHRASE']
            missing_vars = []
            
            with open(env_file, 'r') as f:
                env_content = f.read()
                
                for var in required_vars:
                    if f"{var}=your_" in env_content or f"{var}=" not in env_content:
                        missing_vars.append(var)
            
            if missing_vars:
                return False, f"Missing or default environment variables: {', '.join(missing_vars)}"
            
            return True, "Configuration files are present and configured"
            
        except Exception as e:
            return False, f"Configuration check failed: {str(e)}"
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        checks = [
            ('docker_services', self.check_docker_services),
            ('system_resources', self.check_system_resources),
            ('trading_service', self.check_trading_service),
            ('monitoring_service', self.check_monitoring_service),
            ('redis_service', self.check_redis_service),
            ('api_connectivity', self.check_api_connectivity),
            ('log_files', self.check_log_files),
            ('configuration', self.check_configuration)
        ]
        
        passed = 0
        failed = 0
        
        for check_name, check_func in checks:
            try:
                success, message = check_func()
                
                self.results['checks'][check_name] = {
                    'status': 'PASS' if success else 'FAIL',
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                }
                
                if success:
                    passed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                self.results['checks'][check_name] = {
                    'status': 'ERROR',
                    'message': f"Check failed with exception: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                }
                failed += 1
        
        # Determine overall status
        if failed == 0:
            self.results['overall_status'] = 'HEALTHY'
        elif passed > failed:
            self.results['overall_status'] = 'DEGRADED'
        else:
            self.results['overall_status'] = 'UNHEALTHY'
        
        self.results['summary'] = {
            'total_checks': len(checks),
            'passed': passed,
            'failed': failed
        }
        
        return self.results
    
    def print_results(self):
        """Print health check results in a readable format."""
        print(f"\n{'='*60}")
        print(f"AlphaPulse-RL System Health Check")
        print(f"{'='*60}")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"Summary: {self.results['summary']['passed']}/{self.results['summary']['total_checks']} checks passed")
        print(f"{'='*60}")
        
        for check_name, check_result in self.results['checks'].items():
            status_symbol = "✓" if check_result['status'] == 'PASS' else "✗"
            print(f"{status_symbol} {check_name.replace('_', ' ').title()}: {check_result['status']}")
            print(f"   {check_result['message']}")
            print()
        
        if self.results['overall_status'] != 'HEALTHY':
            print("⚠️  System is not fully healthy. Please address the failed checks above.")
        else:
            print("✅ System is healthy and ready for operation.")
        
        print(f"{'='*60}\n")

def main():
    """Main function."""
    checker = HealthChecker()
    results = checker.run_all_checks()
    
    # Print results to console
    checker.print_results()
    
    # Save results to file
    output_file = f"logs/health_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")
    
    # Exit with appropriate code
    if results['overall_status'] == 'HEALTHY':
        sys.exit(0)
    elif results['overall_status'] == 'DEGRADED':
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == '__main__':
    main()
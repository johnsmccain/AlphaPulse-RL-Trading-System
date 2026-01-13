#!/bin/bash

# AlphaPulse-RL Environment Setup Script
# This script sets up the production environment with all necessary dependencies

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [[ -f /etc/debian_version ]]; then
            OS="debian"
        elif [[ -f /etc/redhat-release ]]; then
            OS="redhat"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        OS="unknown"
    fi
    
    log "Detected OS: $OS"
}

# Install Docker
install_docker() {
    log "Installing Docker..."
    
    case $OS in
        "debian")
            # Update package index
            sudo apt-get update
            
            # Install prerequisites
            sudo apt-get install -y \
                apt-transport-https \
                ca-certificates \
                curl \
                gnupg \
                lsb-release
            
            # Add Docker's official GPG key
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
            
            # Set up stable repository
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            
            # Install Docker Engine
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io
            ;;
        "redhat")
            # Install Docker using yum
            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo yum install -y docker-ce docker-ce-cli containerd.io
            ;;
        "macos")
            if command -v brew &> /dev/null; then
                brew install --cask docker
            else
                error "Homebrew not found. Please install Docker Desktop manually from https://www.docker.com/products/docker-desktop"
                exit 1
            fi
            ;;
        *)
            error "Unsupported OS for automatic Docker installation"
            exit 1
            ;;
    esac
    
    # Start and enable Docker service (Linux only)
    if [[ "$OS" != "macos" ]]; then
        sudo systemctl start docker
        sudo systemctl enable docker
        
        # Add current user to docker group
        sudo usermod -aG docker $USER
        warning "Please log out and log back in for Docker group changes to take effect"
    fi
    
    success "Docker installed successfully"
}

# Install Docker Compose
install_docker_compose() {
    log "Installing Docker Compose..."
    
    # Get latest version
    DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
    
    case $OS in
        "debian"|"redhat"|"linux")
            # Download and install Docker Compose
            sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
            ;;
        "macos")
            if command -v brew &> /dev/null; then
                brew install docker-compose
            else
                # Docker Desktop for Mac includes Docker Compose
                log "Docker Compose should be included with Docker Desktop"
            fi
            ;;
    esac
    
    success "Docker Compose installed successfully"
}

# Install Python dependencies
install_python_deps() {
    log "Installing Python dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Check if Python 3.11+ is available
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc -l) -eq 1 ]]; then
            PYTHON_CMD="python3"
        else
            error "Python 3.9+ is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        error "Python 3 is not installed"
        exit 1
    fi
    
    # Install pip if not available
    if ! command -v pip3 &> /dev/null; then
        case $OS in
            "debian")
                sudo apt-get install -y python3-pip
                ;;
            "redhat")
                sudo yum install -y python3-pip
                ;;
            "macos")
                if command -v brew &> /dev/null; then
                    brew install python
                fi
                ;;
        esac
    fi
    
    # Upgrade pip
    $PYTHON_CMD -m pip install --upgrade pip
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        $PYTHON_CMD -m pip install -r requirements.txt
    else
        warning "requirements.txt not found, skipping Python dependencies"
    fi
    
    success "Python dependencies installed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    cd "$PROJECT_ROOT"
    
    # Create data directories
    mkdir -p data/portfolio/backups
    mkdir -p logs/test_episodes
    mkdir -p logs/test_analysis
    mkdir -p logs/test_performance_analysis
    mkdir -p models/saved
    mkdir -p deployment/monitoring/static
    mkdir -p deployment/monitoring/templates
    
    # Set appropriate permissions
    chmod 755 data logs models
    chmod 700 data/portfolio  # Sensitive financial data
    
    success "Directories created"
}

# Setup configuration files
setup_configuration() {
    log "Setting up configuration files..."
    
    cd "$PROJECT_ROOT"
    
    # Create environment file template if it doesn't exist
    if [[ ! -f "deployment/.env" ]]; then
        cat > deployment/.env << 'EOF'
# AlphaPulse-RL Trading System Environment Configuration

# WEEX API Configuration (REQUIRED)
WEEX_API_KEY=your_api_key_here
WEEX_SECRET_KEY=your_secret_key_here
WEEX_PASSPHRASE=your_passphrase_here

# Trading Configuration
TRADING_MODE=paper  # paper or live
INITIAL_BALANCE=1000
MAX_LEVERAGE=12
MAX_POSITION_SIZE_PERCENT=10
MAX_DAILY_LOSS_PERCENT=3
MAX_TOTAL_DRAWDOWN_PERCENT=12

# System Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
ENABLE_MONITORING=true
MONITORING_INTERVAL=30

# Performance Configuration
CACHE_ENABLED=true
BATCH_PROCESSING_ENABLED=true
PERFORMANCE_MONITORING=true

# Redis Configuration (for caching)
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
EOF
        
        warning "Environment file created at deployment/.env"
        warning "Please update the WEEX API credentials before deployment"
    fi
    
    # Create monitoring configuration
    mkdir -p deployment/monitoring
    cat > deployment/monitoring/requirements.txt << 'EOF'
flask==2.3.3
redis==4.6.0
psutil==5.9.5
requests==2.31.0
plotly==5.17.0
pandas==2.1.1
numpy==1.24.3
gunicorn==21.2.0
EOF
    
    success "Configuration files set up"
}

# Setup monitoring service
setup_monitoring() {
    log "Setting up monitoring service..."
    
    mkdir -p deployment/monitoring
    
    # Create monitoring service
    cat > deployment/monitoring/monitor_service.py << 'EOF'
#!/usr/bin/env python3
"""
AlphaPulse-RL Monitoring Service

Provides real-time monitoring dashboard and alerting for the trading system.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from flask import Flask, render_template, jsonify, request
import redis
import psutil
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
MONITORING_INTERVAL = int(os.getenv('MONITORING_INTERVAL', 30))

# Initialize Redis connection
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    logger.info("Connected to Redis")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'alphapulse-monitor'
    })

@app.route('/')
def dashboard():
    """Main monitoring dashboard."""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """Get current system metrics."""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3)
            }
        }
        
        # Try to get trading system metrics
        try:
            response = requests.get('http://alphapulse-trading:8080/metrics', timeout=5)
            if response.status_code == 200:
                metrics['trading'] = response.json()
        except Exception as e:
            logger.warning(f"Failed to get trading metrics: {e}")
            metrics['trading'] = {'status': 'unavailable'}
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs')
def get_logs():
    """Get recent log entries."""
    try:
        log_entries = []
        log_file = '/app/logs/alphapulse.log'
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Get last 100 lines
                for line in lines[-100:]:
                    if line.strip():
                        log_entries.append({
                            'timestamp': datetime.now().isoformat(),
                            'message': line.strip()
                        })
        
        return jsonify({'logs': log_entries})
        
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=False)
EOF
    
    # Create dashboard template
    mkdir -p deployment/monitoring/templates
    cat > deployment/monitoring/templates/dashboard.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>AlphaPulse-RL Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .metric-label { color: #7f8c8d; margin-bottom: 10px; }
        .status-good { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-error { color: #e74c3c; }
        .logs-container { background: white; padding: 20px; border-radius: 5px; margin-top: 20px; max-height: 400px; overflow-y: auto; }
        .log-entry { font-family: monospace; font-size: 12px; margin-bottom: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AlphaPulse-RL Trading System</h1>
            <p>Real-time Monitoring Dashboard</p>
        </div>
        
        <div class="metrics-grid" id="metrics-grid">
            <!-- Metrics will be loaded here -->
        </div>
        
        <div class="logs-container">
            <h3>Recent Logs</h3>
            <div id="logs-content">
                <!-- Logs will be loaded here -->
            </div>
        </div>
    </div>
    
    <script>
        function updateMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    const grid = document.getElementById('metrics-grid');
                    grid.innerHTML = '';
                    
                    // System metrics
                    if (data.system) {
                        grid.innerHTML += createMetricCard('CPU Usage', data.system.cpu_percent + '%', getStatusClass(data.system.cpu_percent, 80, 90));
                        grid.innerHTML += createMetricCard('Memory Usage', data.system.memory_percent.toFixed(1) + '%', getStatusClass(data.system.memory_percent, 80, 90));
                        grid.innerHTML += createMetricCard('Disk Usage', data.system.disk_percent.toFixed(1) + '%', getStatusClass(data.system.disk_percent, 80, 90));
                    }
                    
                    // Trading metrics
                    if (data.trading && data.trading.status !== 'unavailable') {
                        grid.innerHTML += createMetricCard('Trading Status', data.trading.status || 'Unknown', 'status-good');
                        if (data.trading.portfolio) {
                            grid.innerHTML += createMetricCard('Portfolio Value', '$' + (data.trading.portfolio.total_equity || 0).toFixed(2), 'status-good');
                            grid.innerHTML += createMetricCard('Daily P&L', '$' + (data.trading.portfolio.daily_pnl || 0).toFixed(2), data.trading.portfolio.daily_pnl >= 0 ? 'status-good' : 'status-error');
                        }
                    } else {
                        grid.innerHTML += createMetricCard('Trading Status', 'Unavailable', 'status-error');
                    }
                })
                .catch(error => console.error('Error fetching metrics:', error));
        }
        
        function updateLogs() {
            fetch('/api/logs')
                .then(response => response.json())
                .then(data => {
                    const logsContent = document.getElementById('logs-content');
                    logsContent.innerHTML = '';
                    
                    if (data.logs) {
                        data.logs.forEach(log => {
                            logsContent.innerHTML += '<div class="log-entry">' + log.message + '</div>';
                        });
                        logsContent.scrollTop = logsContent.scrollHeight;
                    }
                })
                .catch(error => console.error('Error fetching logs:', error));
        }
        
        function createMetricCard(label, value, statusClass) {
            return `
                <div class="metric-card">
                    <div class="metric-label">${label}</div>
                    <div class="metric-value ${statusClass}">${value}</div>
                </div>
            `;
        }
        
        function getStatusClass(value, warning, error) {
            if (value >= error) return 'status-error';
            if (value >= warning) return 'status-warning';
            return 'status-good';
        }
        
        // Update every 30 seconds
        updateMetrics();
        updateLogs();
        setInterval(updateMetrics, 30000);
        setInterval(updateLogs, 60000);
    </script>
</body>
</html>
EOF
    
    success "Monitoring service set up"
}

# Setup systemd service (Linux only)
setup_systemd_service() {
    if [[ "$OS" != "debian" && "$OS" != "redhat" ]]; then
        log "Skipping systemd service setup (not on Linux)"
        return
    fi
    
    log "Setting up systemd service..."
    
    # Create systemd service file
    sudo tee /etc/systemd/system/alphapulse-trading.service > /dev/null << EOF
[Unit]
Description=AlphaPulse-RL Trading System
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PROJECT_ROOT/deployment/docker
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
User=$USER

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable alphapulse-trading.service
    
    success "Systemd service configured"
}

# Main setup function
main() {
    log "Starting AlphaPulse-RL environment setup..."
    
    detect_os
    
    # Check if Docker is already installed
    if ! command -v docker &> /dev/null; then
        install_docker
    else
        success "Docker is already installed"
    fi
    
    # Check if Docker Compose is already installed
    if ! command -v docker-compose &> /dev/null; then
        install_docker_compose
    else
        success "Docker Compose is already installed"
    fi
    
    install_python_deps
    create_directories
    setup_configuration
    setup_monitoring
    setup_systemd_service
    
    success "Environment setup completed successfully!"
    
    log ""
    log "Next steps:"
    log "1. Update API credentials in deployment/.env"
    log "2. Review configuration files in config/"
    log "3. Run deployment: ./deployment/scripts/deploy.sh"
    log ""
    log "For manual testing:"
    log "  cd deployment/docker && docker-compose up -d"
}

# Handle command line arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "docker-only")
        detect_os
        install_docker
        install_docker_compose
        ;;
    "python-only")
        install_python_deps
        ;;
    "directories")
        create_directories
        ;;
    "config")
        setup_configuration
        ;;
    "monitoring")
        setup_monitoring
        ;;
    *)
        echo "Usage: $0 {setup|docker-only|python-only|directories|config|monitoring}"
        exit 1
        ;;
esac
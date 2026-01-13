#!/bin/bash

# AlphaPulse-RL Systemd Services Installation Script
# This script installs and configures systemd services for production deployment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SERVICE_USER="${SERVICE_USER:-alphapulse}"

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

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Create service user
create_service_user() {
    log "Creating service user: $SERVICE_USER"
    
    if id "$SERVICE_USER" &>/dev/null; then
        warning "User $SERVICE_USER already exists"
    else
        useradd -r -s /bin/bash -d "$PROJECT_ROOT" -c "AlphaPulse-RL Service User" "$SERVICE_USER"
        success "Created service user: $SERVICE_USER"
    fi
    
    # Add user to docker group
    usermod -aG docker "$SERVICE_USER"
    
    # Set ownership of project directory
    chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_ROOT"
    chmod 755 "$PROJECT_ROOT"
}

# Install main trading service
install_trading_service() {
    log "Installing AlphaPulse trading service..."
    
    cat > /etc/systemd/system/alphapulse-trading.service << EOF
[Unit]
Description=AlphaPulse-RL Trading System
Documentation=https://github.com/your-org/alphapulse-rl
After=docker.service network-online.target
Wants=network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$PROJECT_ROOT/deployment/docker
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=COMPOSE_PROJECT_NAME=alphapulse
ExecStartPre=/usr/local/bin/docker-compose -f docker-compose.yml pull --quiet
ExecStart=/usr/local/bin/docker-compose -f docker-compose.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.yml down --timeout 30
ExecReload=/usr/local/bin/docker-compose -f docker-compose.yml restart
TimeoutStartSec=300
TimeoutStopSec=60
Restart=on-failure
RestartSec=10

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_ROOT

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

    success "Trading service installed"
}

# Install backup service and timer
install_backup_service() {
    log "Installing backup service and timer..."
    
    # Backup service
    cat > /etc/systemd/system/alphapulse-backup.service << EOF
[Unit]
Description=AlphaPulse-RL System Backup
Documentation=https://github.com/your-org/alphapulse-rl

[Service]
Type=oneshot
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/bin/python3 $PROJECT_ROOT/deployment/scripts/backup_system.py create --type data
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_ROOT

[Install]
WantedBy=multi-user.target
EOF

    # Backup timer
    cat > /etc/systemd/system/alphapulse-backup.timer << EOF
[Unit]
Description=Run AlphaPulse-RL backup every 6 hours
Requires=alphapulse-backup.service

[Timer]
OnCalendar=*-*-* 00,06,12,18:00:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF

    success "Backup service and timer installed"
}

# Install monitoring service
install_monitoring_service() {
    log "Installing monitoring service..."
    
    cat > /etc/systemd/system/alphapulse-monitor.service << EOF
[Unit]
Description=AlphaPulse-RL Monitoring Service
Documentation=https://github.com/your-org/alphapulse-rl
After=alphapulse-trading.service
Wants=alphapulse-trading.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$PROJECT_ROOT/deployment/monitoring
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH=$PROJECT_ROOT
ExecStart=/usr/bin/python3 monitor_service.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Health check
ExecStartPost=/bin/sleep 10
ExecStartPost=/bin/bash -c 'curl -f http://localhost:8081/health || exit 1'

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_ROOT

# Resource limits
LimitNOFILE=1024
LimitNPROC=512

[Install]
WantedBy=multi-user.target
EOF

    success "Monitoring service installed"
}

# Install log rotation configuration
install_log_rotation() {
    log "Installing log rotation configuration..."
    
    cat > /etc/logrotate.d/alphapulse << EOF
$PROJECT_ROOT/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
    postrotate
        # Send HUP signal to restart logging if needed
        systemctl reload alphapulse-trading.service || true
    endscript
}

$PROJECT_ROOT/logs/*.csv {
    weekly
    missingok
    rotate 12
    compress
    delaycompress
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
}

$PROJECT_ROOT/logs/*.json {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
}
EOF

    success "Log rotation configured"
}

# Install health check script
install_health_check() {
    log "Installing health check script..."
    
    cat > /usr/local/bin/alphapulse-health-check << EOF
#!/bin/bash
# AlphaPulse-RL Health Check Script

cd $PROJECT_ROOT
python3 deployment/scripts/health_check.py

# Exit with health check status
exit \$?
EOF

    chmod +x /usr/local/bin/alphapulse-health-check
    
    # Add to cron for regular health checks
    cat > /etc/cron.d/alphapulse-health << EOF
# AlphaPulse-RL Health Check - every 5 minutes
*/5 * * * * $SERVICE_USER /usr/local/bin/alphapulse-health-check > /dev/null 2>&1
EOF

    success "Health check script installed"
}

# Configure firewall
configure_firewall() {
    log "Configuring firewall..."
    
    # Check if ufw is available
    if command -v ufw &> /dev/null; then
        # Allow SSH (be careful not to lock yourself out)
        ufw allow ssh
        
        # Allow HTTP and HTTPS
        ufw allow 80/tcp
        ufw allow 443/tcp
        
        # Allow monitoring ports (restrict to local network)
        ufw allow from 10.0.0.0/8 to any port 8080
        ufw allow from 172.16.0.0/12 to any port 8080
        ufw allow from 192.168.0.0/16 to any port 8080
        ufw allow from 127.0.0.1 to any port 8080
        
        ufw allow from 10.0.0.0/8 to any port 8081
        ufw allow from 172.16.0.0/12 to any port 8081
        ufw allow from 192.168.0.0/16 to any port 8081
        ufw allow from 127.0.0.1 to any port 8081
        
        # Enable firewall if not already enabled
        ufw --force enable
        
        success "Firewall configured"
    else
        warning "UFW not available, skipping firewall configuration"
    fi
}

# Set up system limits
configure_system_limits() {
    log "Configuring system limits..."
    
    # Increase file descriptor limits
    cat >> /etc/security/limits.conf << EOF

# AlphaPulse-RL system limits
$SERVICE_USER soft nofile 65536
$SERVICE_USER hard nofile 65536
$SERVICE_USER soft nproc 4096
$SERVICE_USER hard nproc 4096
EOF

    # Configure systemd limits
    mkdir -p /etc/systemd/system.conf.d
    cat > /etc/systemd/system.conf.d/alphapulse.conf << EOF
[Manager]
DefaultLimitNOFILE=65536
DefaultLimitNPROC=4096
EOF

    success "System limits configured"
}

# Enable and start services
enable_services() {
    log "Enabling and starting services..."
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable services
    systemctl enable alphapulse-trading.service
    systemctl enable alphapulse-backup.service
    systemctl enable alphapulse-backup.timer
    systemctl enable alphapulse-monitor.service
    
    # Start timer (service will be started by timer)
    systemctl start alphapulse-backup.timer
    
    success "Services enabled"
    
    log "To start the trading system, run:"
    log "  sudo systemctl start alphapulse-trading.service"
    log "  sudo systemctl start alphapulse-monitor.service"
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    
    # Check service files
    services=("alphapulse-trading" "alphapulse-backup" "alphapulse-monitor")
    
    for service in "${services[@]}"; do
        if systemctl list-unit-files | grep -q "$service.service"; then
            success "Service $service.service is installed"
        else
            error "Service $service.service is not installed"
        fi
    done
    
    # Check timer
    if systemctl list-unit-files | grep -q "alphapulse-backup.timer"; then
        success "Timer alphapulse-backup.timer is installed"
    else
        error "Timer alphapulse-backup.timer is not installed"
    fi
    
    # Check user
    if id "$SERVICE_USER" &>/dev/null; then
        success "Service user $SERVICE_USER exists"
    else
        error "Service user $SERVICE_USER does not exist"
    fi
    
    # Check permissions
    if [[ -r "$PROJECT_ROOT" ]]; then
        success "Project directory is accessible"
    else
        error "Project directory is not accessible"
    fi
}

# Main installation function
main() {
    log "Starting AlphaPulse-RL systemd services installation..."
    
    check_root
    create_service_user
    install_trading_service
    install_backup_service
    install_monitoring_service
    install_log_rotation
    install_health_check
    configure_firewall
    configure_system_limits
    enable_services
    verify_installation
    
    success "Installation completed successfully!"
    
    log ""
    log "Next steps:"
    log "1. Configure environment variables in $PROJECT_ROOT/deployment/.env"
    log "2. Start services: sudo systemctl start alphapulse-trading.service"
    log "3. Check status: sudo systemctl status alphapulse-trading.service"
    log "4. View logs: sudo journalctl -u alphapulse-trading.service -f"
    log ""
    log "Service management commands:"
    log "  sudo systemctl start alphapulse-trading.service"
    log "  sudo systemctl stop alphapulse-trading.service"
    log "  sudo systemctl restart alphapulse-trading.service"
    log "  sudo systemctl status alphapulse-trading.service"
    log ""
    log "Health check: /usr/local/bin/alphapulse-health-check"
}

# Handle command line arguments
case "${1:-install}" in
    "install")
        main
        ;;
    "uninstall")
        log "Uninstalling services..."
        systemctl stop alphapulse-trading.service alphapulse-monitor.service alphapulse-backup.timer || true
        systemctl disable alphapulse-trading.service alphapulse-backup.service alphapulse-backup.timer alphapulse-monitor.service || true
        rm -f /etc/systemd/system/alphapulse-*.service /etc/systemd/system/alphapulse-*.timer
        rm -f /etc/logrotate.d/alphapulse
        rm -f /usr/local/bin/alphapulse-health-check
        rm -f /etc/cron.d/alphapulse-health
        systemctl daemon-reload
        success "Services uninstalled"
        ;;
    "verify")
        verify_installation
        ;;
    *)
        echo "Usage: $0 {install|uninstall|verify}"
        echo ""
        echo "Commands:"
        echo "  install   - Install systemd services (default)"
        echo "  uninstall - Remove systemd services"
        echo "  verify    - Verify installation"
        exit 1
        ;;
esac
#!/bin/bash

# AlphaPulse-RL Trading System Deployment Script
# This script handles production deployment with safety checks and rollback capabilities

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"
BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "$DEPLOYMENT_DIR/.env" ]]; then
        error "Environment file not found at $DEPLOYMENT_DIR/.env"
        error "Please create the environment file with required API keys"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Validate environment configuration
validate_environment() {
    log "Validating environment configuration..."
    
    # Source environment file
    source "$DEPLOYMENT_DIR/.env"
    
    # Check required environment variables
    required_vars=("WEEX_API_KEY" "WEEX_SECRET_KEY" "WEEX_PASSPHRASE")
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Validate API key format (basic check)
    if [[ ${#WEEX_API_KEY} -lt 20 ]]; then
        warning "API key seems too short, please verify"
    fi
    
    success "Environment validation passed"
}

# Create backup of current deployment
create_backup() {
    log "Creating backup of current deployment..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup configuration
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        cp -r "$PROJECT_ROOT/config" "$BACKUP_DIR/"
    fi
    
    # Backup data
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        cp -r "$PROJECT_ROOT/data" "$BACKUP_DIR/"
    fi
    
    # Backup logs (last 7 days)
    if [[ -d "$PROJECT_ROOT/logs" ]]; then
        mkdir -p "$BACKUP_DIR/logs"
        find "$PROJECT_ROOT/logs" -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/logs/" \;
    fi
    
    # Backup models
    if [[ -d "$PROJECT_ROOT/models" ]]; then
        cp -r "$PROJECT_ROOT/models" "$BACKUP_DIR/"
    fi
    
    success "Backup created at $BACKUP_DIR"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$DEPLOYMENT_DIR/docker"
    
    # Build main trading application
    docker-compose build --no-cache alphapulse-trading
    
    # Build monitoring service
    docker-compose build --no-cache alphapulse-monitor
    
    success "Docker images built successfully"
}

# Run pre-deployment tests
run_tests() {
    log "Running pre-deployment tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run basic system tests
    if [[ -f "test_system_integration.py" ]]; then
        python test_system_integration.py --quick
    fi
    
    # Test configuration loading
    python -c "
import yaml
import sys
try:
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('Configuration file is valid')
except Exception as e:
    print(f'Configuration file error: {e}')
    sys.exit(1)
"
    
    success "Pre-deployment tests passed"
}

# Deploy the application
deploy_application() {
    log "Deploying AlphaPulse-RL Trading System..."
    
    cd "$DEPLOYMENT_DIR/docker"
    
    # Stop existing containers gracefully
    docker-compose down --timeout 30
    
    # Remove old containers and networks
    docker-compose rm -f
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if docker-compose ps | grep -q "healthy"; then
            success "Services are healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            error "Services failed to become healthy within timeout"
            return 1
        fi
        
        log "Attempt $attempt/$max_attempts - waiting for services..."
        sleep 10
        ((attempt++))
    done
    
    success "Application deployed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    cd "$DEPLOYMENT_DIR/docker"
    
    # Check container status
    if ! docker-compose ps | grep -q "Up"; then
        error "Some containers are not running"
        docker-compose logs --tail=50
        return 1
    fi
    
    # Check health endpoints
    local health_checks=("http://localhost:8080/health" "http://localhost:8081/health")
    
    for endpoint in "${health_checks[@]}"; do
        if curl -f "$endpoint" &> /dev/null; then
            success "Health check passed for $endpoint"
        else
            warning "Health check failed for $endpoint"
        fi
    done
    
    # Check logs for errors
    if docker-compose logs --tail=100 | grep -i error; then
        warning "Errors found in logs, please review"
    fi
    
    success "Deployment verification completed"
}

# Rollback function
rollback() {
    error "Deployment failed, initiating rollback..."
    
    cd "$DEPLOYMENT_DIR/docker"
    
    # Stop current deployment
    docker-compose down --timeout 30
    
    # Restore from backup
    if [[ -d "$BACKUP_DIR" ]]; then
        log "Restoring from backup..."
        
        # Restore configuration
        if [[ -d "$BACKUP_DIR/config" ]]; then
            rm -rf "$PROJECT_ROOT/config"
            cp -r "$BACKUP_DIR/config" "$PROJECT_ROOT/"
        fi
        
        # Restore data
        if [[ -d "$BACKUP_DIR/data" ]]; then
            rm -rf "$PROJECT_ROOT/data"
            cp -r "$BACKUP_DIR/data" "$PROJECT_ROOT/"
        fi
        
        success "Rollback completed"
    else
        error "No backup found for rollback"
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Remove old Docker images
    docker image prune -f
    
    # Remove old backups (keep last 5)
    if [[ -d "$PROJECT_ROOT/backups" ]]; then
        cd "$PROJECT_ROOT/backups"
        ls -t | tail -n +6 | xargs -r rm -rf
    fi
    
    success "Cleanup completed"
}

# Main deployment function
main() {
    log "Starting AlphaPulse-RL Trading System deployment..."
    
    # Trap errors for rollback
    trap rollback ERR
    
    check_root
    check_prerequisites
    validate_environment
    create_backup
    build_images
    run_tests
    deploy_application
    verify_deployment
    cleanup
    
    success "Deployment completed successfully!"
    
    log "Services are running:"
    log "  - Trading System: http://localhost:8080"
    log "  - Monitoring Dashboard: http://localhost:8081"
    log ""
    log "To view logs: docker-compose -f $DEPLOYMENT_DIR/docker/docker-compose.yml logs -f"
    log "To stop services: docker-compose -f $DEPLOYMENT_DIR/docker/docker-compose.yml down"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback
        ;;
    "status")
        cd "$DEPLOYMENT_DIR/docker"
        docker-compose ps
        ;;
    "logs")
        cd "$DEPLOYMENT_DIR/docker"
        docker-compose logs -f "${2:-}"
        ;;
    "stop")
        cd "$DEPLOYMENT_DIR/docker"
        docker-compose down --timeout 30
        ;;
    "restart")
        cd "$DEPLOYMENT_DIR/docker"
        docker-compose restart "${2:-}"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status|logs|stop|restart}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the application (default)"
        echo "  rollback - Rollback to previous version"
        echo "  status   - Show container status"
        echo "  logs     - Show container logs"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart services"
        exit 1
        ;;
esac
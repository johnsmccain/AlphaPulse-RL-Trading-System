# AlphaPulse-RL Deployment Guide

## Quick Start

This directory contains all the necessary files and scripts for deploying the AlphaPulse-RL trading system in production.

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+
- 4+ CPU cores, 8GB+ RAM, 100GB+ SSD

### Deployment Steps

1. **Environment Setup**
   ```bash
   ./scripts/setup_environment.sh
   ```

2. **Configuration**
   ```bash
   cp .env.template .env
   # Edit .env with your WEEX API credentials
   ```

3. **Finalize Deployment**
   ```bash
   ./scripts/finalize_deployment.py
   ```

4. **Deploy System**
   ```bash
   ./scripts/deploy.sh
   ```

5. **Validate Deployment**
   ```bash
   ./scripts/validate_deployment.py
   ./scripts/health_check.py
   ```

## Directory Structure

```
deployment/
├── README.md                           # This file
├── .env.template                       # Environment variables template
├── production_config.yaml             # Production configuration
├── production_checklist.md            # Deployment checklist
├── PRODUCTION_DEPLOYMENT_GUIDE.md     # Comprehensive deployment guide
├── SYSTEM_OPERATIONS_MANUAL.md        # Operations manual
├── DEPLOYMENT_SUMMARY.md              # Deployment summary (generated)
├── docker/
│   ├── docker-compose.yml             # Service orchestration
│   ├── Dockerfile                     # Main application container
│   ├── Dockerfile.monitor             # Monitoring container
│   ├── nginx.conf                     # Nginx configuration
│   ├── fluentd.conf                   # Log aggregation config
│   └── ssl/                           # SSL certificates directory
├── monitoring/
│   ├── monitor_service.py             # Monitoring service
│   ├── alerting_rules.py              # Alert rules and thresholds
│   ├── templates/
│   │   └── dashboard.html             # Monitoring dashboard
│   └── static/                        # Dashboard assets
├── scripts/
│   ├── setup_environment.sh           # Environment setup
│   ├── deploy.sh                      # Main deployment script
│   ├── finalize_deployment.py         # Deployment finalization
│   ├── validate_deployment.py         # Deployment validation
│   ├── health_check.py                # System health check
│   └── backup_system.py               # Backup management
└── systemd/
    ├── alphapulse-trading.service     # Main systemd service
    ├── alphapulse-backup.service      # Backup service
    └── alphapulse-backup.timer        # Backup timer
```

## Key Components

### Core Services

- **alphapulse-trading**: Main trading application
- **alphapulse-monitor**: Real-time monitoring dashboard
- **alphapulse-redis**: Caching and session storage
- **alphapulse-proxy**: Nginx reverse proxy
- **alphapulse-logs**: Log aggregation with Fluentd

### Configuration Files

- **`.env`**: Environment variables (API keys, trading parameters)
- **`production_config.yaml`**: Production-specific settings
- **`docker-compose.yml`**: Service orchestration
- **`nginx.conf`**: Web server and proxy configuration

### Scripts

- **`setup_environment.sh`**: Installs dependencies and sets up environment
- **`deploy.sh`**: Main deployment script with rollback capability
- **`finalize_deployment.py`**: Final preparation and validation
- **`validate_deployment.py`**: Comprehensive deployment validation
- **`health_check.py`**: System health monitoring
- **`backup_system.py`**: Automated backup management

## Monitoring and Alerting

### Dashboard Access

- **URL**: http://localhost:8081 (or configured domain)
- **Features**: Real-time metrics, alerts, logs, performance analytics

### Key Metrics

- System resources (CPU, Memory, Disk)
- Trading performance (P&L, Sharpe ratio, drawdown)
- Risk metrics (position sizes, leverage, limits)
- API connectivity and response times
- Model inference performance

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|---------|
| CPU Usage | 80% | 90% | Scale resources |
| Memory Usage | 80% | 90% | Scale resources |
| Disk Usage | 80% | 90% | Clean logs/data |
| Daily Loss | 2.5% | 3% | Review strategy |
| Total Drawdown | 10% | 12% | Emergency stop |
| API Errors | 10% | 25% | Check connectivity |

## Security Features

- SSL/TLS encryption for web interfaces
- API key rotation and secure storage
- Firewall configuration and IP whitelisting
- HTTP basic authentication for monitoring
- Audit logging and access control
- Data encryption at rest

## Backup Strategy

### Automated Backups

- **Daily**: Incremental data backups (2:00 AM)
- **Weekly**: Full system backups (Sunday 1:00 AM)
- **Monthly**: Configuration and model backups

### Backup Components

- Trading data and logs
- Model files and configurations
- System configurations
- Database snapshots (if applicable)

### Retention Policy

- Daily backups: 30 days
- Weekly backups: 12 weeks
- Monthly backups: 12 months

## Operational Procedures

### Daily Operations

1. Check monitoring dashboard for alerts
2. Review daily P&L and risk metrics
3. Verify system health and resource usage
4. Check trade logs for anomalies
5. Confirm API connectivity is stable

### Weekly Operations

1. Analyze weekly performance metrics
2. Review system resource usage trends
3. Check log file sizes and rotation
4. Verify backup completion
5. Update system packages if needed

### Emergency Procedures

#### System Down
```bash
# Check service status
./scripts/deploy.sh status

# Review logs
./scripts/deploy.sh logs

# Restart services
./scripts/deploy.sh restart

# Rollback if needed
./scripts/deploy.sh rollback
```

#### High Risk Alert
1. Check monitoring dashboard immediately
2. Review recent trades and market conditions
3. Consider reducing position sizes or stopping trading
4. Document incident and response actions

#### API Issues
1. Check WEEX exchange status
2. Verify API credentials and permissions
3. Check network connectivity and rate limits
4. Contact exchange support if needed

## Performance Optimization

### System Tuning

- Docker resource limits optimization
- Redis memory configuration
- Log level adjustment for production
- Caching configuration

### Application Tuning

- Model inference performance profiling
- Data processing pipeline optimization
- Memory usage pattern analysis
- Database query optimization

## Troubleshooting

### Common Issues

1. **Container won't start**: Check logs, verify configuration, restart Docker
2. **High resource usage**: Check processes, restart services, scale resources
3. **API connection errors**: Verify credentials, check connectivity, review rate limits
4. **Poor trading performance**: Check model, review market conditions, analyze trades

### Log Analysis

```bash
# View recent logs
tail -f logs/alphapulse_$(date +%Y%m%d).log

# Check service status
docker-compose -f docker/docker-compose.yml ps

# View system metrics
curl http://localhost:8081/api/metrics

# Check alerts
curl http://localhost:8081/api/alerts
```

## Support and Documentation

### Additional Documentation

- **`PRODUCTION_DEPLOYMENT_GUIDE.md`**: Comprehensive deployment guide
- **`SYSTEM_OPERATIONS_MANUAL.md`**: Detailed operations manual
- **`production_checklist.md`**: Pre-deployment checklist
- **`DEPLOYMENT_SUMMARY.md`**: Post-deployment summary

### Getting Help

1. Check the monitoring dashboard for system status
2. Review log files for error messages
3. Consult the operations manual for procedures
4. Run health checks and validation scripts
5. Contact system administrator or development team

## Version Information

- **System Version**: Production Ready
- **Docker Compose Version**: 3.8
- **Python Version**: 3.9+
- **Deployment Method**: Docker Compose with systemd integration

---

For detailed deployment instructions, see `PRODUCTION_DEPLOYMENT_GUIDE.md`.
For operational procedures, see `SYSTEM_OPERATIONS_MANUAL.md`.
For pre-deployment validation, see `production_checklist.md`.
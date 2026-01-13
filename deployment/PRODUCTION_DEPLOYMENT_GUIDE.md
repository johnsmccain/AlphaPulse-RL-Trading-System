# AlphaPulse-RL Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the AlphaPulse-RL trading system in a production environment with proper monitoring, alerting, and operational procedures.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ or CentOS 8+)
- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **Memory**: Minimum 8GB RAM, Recommended 16GB+ RAM
- **Storage**: Minimum 100GB SSD, Recommended 500GB+ SSD
- **Network**: Stable internet connection with low latency to WEEX exchange

### Software Dependencies

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+
- Git
- SSL certificates (for HTTPS)

## Pre-Deployment Setup

### 1. Environment Preparation

Run the environment setup script:

```bash
./deployment/scripts/setup_environment.sh
```

This script will:
- Install Docker and Docker Compose
- Set up Python dependencies
- Create necessary directories
- Configure monitoring services
- Set up systemd services (Linux only)

### 2. Configuration

#### API Credentials

Edit `deployment/.env` and configure your WEEX API credentials:

```bash
# WEEX API Configuration (REQUIRED)
WEEX_API_KEY=your_actual_api_key_here
WEEX_SECRET_KEY=your_actual_secret_key_here
WEEX_PASSPHRASE=your_actual_passphrase_here

# Trading Configuration
TRADING_MODE=paper  # Start with paper trading
INITIAL_BALANCE=1000
MAX_LEVERAGE=12
MAX_POSITION_SIZE_PERCENT=10
MAX_DAILY_LOSS_PERCENT=3
MAX_TOTAL_DRAWDOWN_PERCENT=12
```

#### SSL Certificates

For production HTTPS access, place your SSL certificates in `deployment/docker/ssl/`:

```bash
mkdir -p deployment/docker/ssl
cp your_cert.pem deployment/docker/ssl/cert.pem
cp your_key.pem deployment/docker/ssl/key.pem
```

#### Monitoring Authentication

Create HTTP basic auth for monitoring dashboard:

```bash
# Install htpasswd utility
sudo apt-get install apache2-utils

# Create password file
htpasswd -c deployment/docker/.htpasswd admin
```

#### Alerting Configuration

Configure alerting in `deployment/.env`:

```bash
# Email alerts (optional)
ALERT_EMAIL=admin@yourcompany.com

# Webhook alerts (optional - Slack, Discord, etc.)
ALERT_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### 3. Network Configuration

#### Firewall Rules

Configure firewall to allow necessary ports:

```bash
# Allow SSH
sudo ufw allow 22

# Allow HTTP/HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Allow monitoring (restrict to your IP)
sudo ufw allow from YOUR_IP_ADDRESS to any port 8080
sudo ufw allow from YOUR_IP_ADDRESS to any port 8081

# Enable firewall
sudo ufw enable
```

#### DNS Configuration

Set up DNS records for your monitoring dashboard:

```
api.alphapulse.yourdomain.com -> Your Server IP
monitor.alphapulse.yourdomain.com -> Your Server IP
```

## Deployment Process

### 1. Deploy the System

Run the deployment script:

```bash
./deployment/scripts/deploy.sh
```

The deployment script will:
- Validate prerequisites and configuration
- Create backups of existing deployment
- Build Docker images
- Run pre-deployment tests
- Deploy all services
- Verify deployment health
- Provide rollback capability if needed

### 2. Verify Deployment

Check service status:

```bash
./deployment/scripts/deploy.sh status
```

View logs:

```bash
./deployment/scripts/deploy.sh logs
```

Access monitoring dashboard:
- https://monitor.alphapulse.yourdomain.com (or http://localhost:8081)

### 3. Initial Testing

#### Paper Trading Mode

Start with paper trading to verify system functionality:

1. Ensure `TRADING_MODE=paper` in `.env`
2. Monitor the dashboard for 24 hours
3. Verify all metrics are updating correctly
4. Check that risk limits are being enforced
5. Review trade logs and AI decision logs

#### Live Trading Transition

Only after successful paper trading:

1. Update `TRADING_MODE=live` in `.env`
2. Restart services: `./deployment/scripts/deploy.sh restart`
3. Monitor closely for the first few hours
4. Verify real trades are being executed correctly

## Monitoring and Alerting

### Dashboard Access

The monitoring dashboard provides real-time visibility into:

- **System Metrics**: CPU, memory, disk usage
- **Trading Metrics**: Portfolio value, P&L, drawdown, trade count
- **Performance Metrics**: Sharpe ratio, win rate, risk metrics
- **Alerts**: Critical system and trading alerts
- **Logs**: Real-time system and trading logs

### Alert Thresholds

The system monitors and alerts on:

| Metric | Warning | Critical | Action |
|--------|---------|----------|---------|
| CPU Usage | 80% | 90% | Scale resources |
| Memory Usage | 80% | 90% | Scale resources |
| Disk Usage | 80% | 90% | Clean logs/data |
| Daily Loss | 2.5% | 3% | Review strategy |
| Total Drawdown | 10% | 12% | Emergency stop |
| API Errors | 5% | 10% | Check connectivity |

### Log Files

Key log files to monitor:

- `logs/alphapulse_YYYYMMDD.log` - Main system logs
- `logs/trades.csv` - Trade execution history
- `logs/ai_decisions.json` - AI decision details
- `logs/risk_alerts.json` - Risk management alerts

## Operational Procedures

### Daily Operations

#### Morning Checklist

1. Check monitoring dashboard for overnight alerts
2. Review trade performance from previous day
3. Verify system health metrics
4. Check log files for any errors or warnings
5. Validate API connectivity and data feeds

#### Evening Checklist

1. Review daily P&L and risk metrics
2. Check system resource usage trends
3. Verify backup completion
4. Review any alerts or incidents
5. Plan any maintenance activities

### Weekly Operations

1. Review weekly performance metrics
2. Analyze trade patterns and strategy effectiveness
3. Update risk parameters if needed
4. Perform system maintenance and updates
5. Review and rotate log files

### Monthly Operations

1. Comprehensive performance review
2. System capacity planning
3. Security updates and patches
4. Backup verification and testing
5. Disaster recovery testing

### Emergency Procedures

#### System Down

1. Check service status: `./deployment/scripts/deploy.sh status`
2. Review logs: `./deployment/scripts/deploy.sh logs`
3. Restart services: `./deployment/scripts/deploy.sh restart`
4. If issues persist, rollback: `./deployment/scripts/deploy.sh rollback`

#### High Drawdown Alert

1. Immediately check monitoring dashboard
2. Review recent trades and market conditions
3. Consider reducing position sizes or stopping trading
4. Analyze risk management effectiveness
5. Document incident and lessons learned

#### API Connectivity Issues

1. Check WEEX exchange status
2. Verify API credentials and permissions
3. Check network connectivity
4. Review rate limiting and IP restrictions
5. Contact exchange support if needed

## Maintenance and Updates

### System Updates

1. Create backup: Automatic during deployment
2. Test updates in staging environment first
3. Schedule maintenance window
4. Deploy updates: `./deployment/scripts/deploy.sh`
5. Verify system functionality
6. Monitor for 24 hours post-update

### Model Updates

1. Train new model version
2. Validate model performance in backtesting
3. Test with paper trading
4. Deploy to production with gradual rollout
5. Monitor performance closely

### Configuration Changes

1. Update configuration files
2. Validate configuration syntax
3. Test changes in staging environment
4. Deploy with backup and rollback plan
5. Monitor system behavior

## Security Considerations

### Access Control

- Use strong passwords for all accounts
- Enable two-factor authentication where possible
- Restrict SSH access to specific IP addresses
- Use VPN for remote access
- Regularly rotate API keys and passwords

### Network Security

- Configure firewall rules properly
- Use HTTPS for all web interfaces
- Implement rate limiting
- Monitor for suspicious activity
- Keep security patches up to date

### Data Protection

- Encrypt sensitive data at rest
- Use secure communication channels
- Implement proper backup encryption
- Follow data retention policies
- Comply with relevant regulations

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check container logs
docker logs alphapulse-trading

# Check resource usage
docker stats

# Restart specific service
docker-compose restart alphapulse-trading
```

#### High Memory Usage

```bash
# Check memory usage by container
docker stats

# Check system memory
free -h

# Restart services to clear memory
./deployment/scripts/deploy.sh restart
```

#### API Connection Errors

```bash
# Test API connectivity
curl -X GET "https://api.weex.com/api/v1/time"

# Check API credentials
grep WEEX_API_KEY deployment/.env

# Review API rate limits
tail -f logs/alphapulse_*.log | grep -i "rate limit"
```

### Performance Issues

#### Slow Response Times

1. Check system resource usage
2. Review database query performance
3. Analyze network latency
4. Consider scaling resources
5. Optimize code if necessary

#### High CPU Usage

1. Identify resource-intensive processes
2. Check for infinite loops or deadlocks
3. Review model inference performance
4. Consider CPU scaling
5. Optimize algorithms if needed

## Support and Escalation

### Internal Support

1. Check this documentation first
2. Review system logs and monitoring
3. Search knowledge base for similar issues
4. Escalate to development team if needed

### External Support

1. WEEX Exchange Support: For API-related issues
2. Cloud Provider Support: For infrastructure issues
3. Security Team: For security incidents
4. Compliance Team: For regulatory issues

## Backup and Recovery

### Automated Backups

The system automatically backs up:
- Configuration files
- Trading data and logs
- Model files
- Database snapshots

### Manual Backup

```bash
# Create manual backup
./deployment/scripts/deploy.sh backup

# Restore from backup
./deployment/scripts/deploy.sh restore BACKUP_DATE
```

### Disaster Recovery

1. Maintain offsite backups
2. Document recovery procedures
3. Test recovery process regularly
4. Maintain emergency contact list
5. Have rollback plan ready

## Performance Optimization

### System Tuning

- Optimize Docker resource limits
- Tune database parameters
- Configure caching appropriately
- Monitor and adjust as needed

### Application Tuning

- Profile model inference performance
- Optimize data processing pipelines
- Implement efficient caching strategies
- Monitor memory usage patterns

## Compliance and Auditing

### Audit Trail

The system maintains comprehensive audit trails:
- All trading decisions and rationale
- Risk management actions
- System configuration changes
- User access and actions

### Regulatory Compliance

- Maintain required records
- Implement proper controls
- Regular compliance reviews
- Document all procedures

## Conclusion

This deployment guide provides the foundation for running AlphaPulse-RL in production. Regular monitoring, maintenance, and adherence to operational procedures are essential for successful operation.

For additional support or questions, refer to the system documentation or contact the development team.
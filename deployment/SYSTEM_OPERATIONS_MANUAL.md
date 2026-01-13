# AlphaPulse-RL System Operations Manual

## Overview

This manual provides comprehensive operational procedures for the AlphaPulse-RL trading system in production environments. It covers daily operations, monitoring, troubleshooting, and emergency procedures.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Daily Operations](#daily-operations)
3. [Monitoring and Alerting](#monitoring-and-alerting)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Emergency Procedures](#emergency-procedures)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Performance Optimization](#performance-optimization)
8. [Security Operations](#security-operations)
9. [Backup and Recovery](#backup-and-recovery)
10. [Incident Response](#incident-response)

## System Architecture Overview

### Core Components

- **Trading System** (`alphapulse-trading`): Main trading application
- **Monitoring Service** (`alphapulse-monitor`): Real-time monitoring and alerting
- **Redis Cache** (`alphapulse-redis`): Caching and session storage
- **Nginx Proxy** (`alphapulse-proxy`): Reverse proxy and load balancer
- **Log Aggregation** (`alphapulse-logs`): Centralized logging with Fluentd

### Service Dependencies

```
Trading System → Redis Cache
Monitoring Service → Trading System, Redis Cache
Nginx Proxy → Trading System, Monitoring Service
Log Aggregation → All Services
```

### Key Directories

- `/opt/alphapulse-rl/`: Main application directory
- `/opt/alphapulse-rl/logs/`: System logs
- `/opt/alphapulse-rl/data/`: Trading data and portfolio state
- `/opt/alphapulse-rl/models/`: ML model files
- `/opt/alphapulse-rl/backups/`: System backups

## Daily Operations

### Morning Checklist (Start of Trading Day)

1. **System Health Check**
   ```bash
   # Check service status
   sudo systemctl status alphapulse-trading
   docker-compose -f /opt/alphapulse-rl/deployment/docker/docker-compose.yml ps
   
   # Check system resources
   htop
   df -h
   ```

2. **Review Overnight Activity**
   ```bash
   # Check recent logs
   tail -100 /opt/alphapulse-rl/logs/alphapulse_$(date +%Y%m%d).log
   
   # Review trades from previous day
   tail -50 /opt/alphapulse-rl/logs/trades.csv
   
   # Check for alerts
   curl -s http://localhost:8081/api/alerts | jq '.active_alerts'
   ```

3. **Verify Trading Configuration**
   ```bash
   # Check trading mode
   grep TRADING_MODE /opt/alphapulse-rl/deployment/.env
   
   # Verify API connectivity
   curl -s http://localhost:8080/health
   ```

4. **Portfolio Status Review**
   ```bash
   # Get current portfolio metrics
   curl -s http://localhost:8080/metrics | jq '.portfolio'
   ```

### Evening Checklist (End of Trading Day)

1. **Performance Review**
   ```bash
   # Generate daily performance report
   python /opt/alphapulse-rl/models/performance_analysis.py --daily
   
   # Review trade statistics
   python -c "
   import pandas as pd
   trades = pd.read_csv('/opt/alphapulse-rl/logs/trades.csv')
   today_trades = trades[trades['timestamp'].str.contains('$(date +%Y-%m-%d)')]
   print(f'Trades today: {len(today_trades)}')
   print(f'Total PnL: {today_trades[\"pnl\"].sum():.2f}')
   print(f'Win rate: {(today_trades[\"pnl\"] > 0).mean():.2%}')
   "
   ```

2. **System Maintenance**
   ```bash
   # Rotate logs if needed
   sudo logrotate /etc/logrotate.d/alphapulse
   
   # Clean up temporary files
   find /opt/alphapulse-rl/data -name "*.tmp" -mtime +1 -delete
   
   # Update system packages (if maintenance window)
   sudo apt update && sudo apt list --upgradable
   ```

3. **Backup Verification**
   ```bash
   # Check backup status
   python /opt/alphapulse-rl/deployment/scripts/backup_system.py list
   
   # Verify latest backup
   ls -la /opt/alphapulse-rl/backups/ | head -5
   ```

### Weekly Operations

1. **Comprehensive Performance Analysis**
   ```bash
   # Generate weekly report
   python /opt/alphapulse-rl/models/performance_analysis.py --weekly
   
   # Analyze model performance
   python /opt/alphapulse-rl/models/evaluate.py --period 7d
   ```

2. **System Health Assessment**
   ```bash
   # Check disk usage trends
   df -h
   du -sh /opt/alphapulse-rl/logs/*
   
   # Review error logs
   grep -i error /opt/alphapulse-rl/logs/alphapulse_*.log | tail -50
   
   # Check memory usage patterns
   free -h
   cat /proc/meminfo
   ```

3. **Security Review**
   ```bash
   # Check authentication logs
   sudo grep "alphapulse" /var/log/auth.log | tail -20
   
   # Review API access patterns
   grep "API" /opt/alphapulse-rl/logs/alphapulse_*.log | tail -50
   ```

## Monitoring and Alerting

### Monitoring Dashboard

Access the monitoring dashboard at: `http://your-server:8081`

**Key Metrics to Monitor:**

- **System Metrics**: CPU, Memory, Disk usage
- **Trading Metrics**: Portfolio value, Daily P&L, Drawdown
- **Performance Metrics**: Trade frequency, Model confidence, API response times
- **Risk Metrics**: Position sizes, Leverage usage, Risk limits

### Alert Categories

#### Critical Alerts (Immediate Action Required)

- **Trading System Down**: System not responding
- **Daily Loss Limit**: Approaching 3% daily loss
- **Drawdown Limit**: Approaching 12% total drawdown
- **High CPU/Memory**: System resources critically low
- **API Failures**: High error rate with exchange

#### Warning Alerts (Monitor Closely)

- **High Resource Usage**: CPU/Memory above 80%
- **Model Confidence Low**: Average confidence below 60%
- **API Errors**: Error rate above 10%
- **Slow Performance**: Model inference taking too long

### Alert Response Procedures

#### Trading System Down
1. Check container status: `docker ps`
2. Review logs: `docker logs alphapulse-trading`
3. Restart if needed: `docker-compose restart alphapulse-trading`
4. Verify recovery: `curl http://localhost:8080/health`

#### High Daily Loss
1. Check current positions: `curl http://localhost:8080/metrics | jq '.portfolio'`
2. Review recent trades: `tail -20 /opt/alphapulse-rl/logs/trades.csv`
3. Consider reducing position sizes or stopping trading
4. Analyze market conditions and strategy performance

#### Resource Alerts
1. Identify resource-intensive processes: `htop`
2. Check disk space: `df -h`
3. Clean up logs if needed: `find /opt/alphapulse-rl/logs -name "*.log" -mtime +7 -delete`
4. Consider scaling resources if persistent

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Trading System Won't Start

**Symptoms**: Container fails to start, health check fails

**Diagnosis**:
```bash
# Check container logs
docker logs alphapulse-trading

# Check configuration
cat /opt/alphapulse-rl/deployment/.env

# Check port availability
netstat -tlnp | grep 8080
```

**Solutions**:
- Verify API credentials are correct
- Check if ports are already in use
- Ensure sufficient disk space
- Restart Docker daemon if needed

#### 2. High Memory Usage

**Symptoms**: Memory usage above 90%, system slowdown

**Diagnosis**:
```bash
# Check memory usage by process
ps aux --sort=-%mem | head -10

# Check container memory usage
docker stats

# Check for memory leaks
cat /proc/meminfo
```

**Solutions**:
- Restart trading system: `docker-compose restart alphapulse-trading`
- Clear Redis cache: `redis-cli FLUSHALL`
- Increase memory limits in docker-compose.yml
- Optimize model inference batch sizes

#### 3. API Connection Issues

**Symptoms**: High API error rates, failed trades

**Diagnosis**:
```bash
# Test API connectivity
curl -X GET "https://api.weex.com/api/v1/time"

# Check API credentials
grep WEEX_API /opt/alphapulse-rl/deployment/.env

# Review API error logs
grep -i "api error" /opt/alphapulse-rl/logs/alphapulse_*.log
```

**Solutions**:
- Verify API credentials and permissions
- Check IP whitelist settings
- Review rate limiting configuration
- Contact exchange support if needed

#### 4. Model Performance Issues

**Symptoms**: Low confidence scores, poor trade performance

**Diagnosis**:
```bash
# Check model metrics
python /opt/alphapulse-rl/models/evaluate.py --quick

# Review recent predictions
tail -50 /opt/alphapulse-rl/logs/ai_decisions.json

# Check data quality
python /opt/alphapulse-rl/data/weex_fetcher.py --test
```

**Solutions**:
- Retrain model with recent data
- Adjust confidence thresholds
- Review feature engineering pipeline
- Check for data quality issues

### Log Analysis

#### Key Log Files

- **Main System Log**: `/opt/alphapulse-rl/logs/alphapulse_YYYYMMDD.log`
- **Trade History**: `/opt/alphapulse-rl/logs/trades.csv`
- **AI Decisions**: `/opt/alphapulse-rl/logs/ai_decisions.json`
- **Risk Alerts**: `/opt/alphapulse-rl/logs/risk_alerts.json`

#### Useful Log Commands

```bash
# Find errors in last 24 hours
find /opt/alphapulse-rl/logs -name "*.log" -mtime -1 -exec grep -i error {} +

# Monitor live logs
tail -f /opt/alphapulse-rl/logs/alphapulse_$(date +%Y%m%d).log

# Search for specific patterns
grep -r "RISK_LIMIT_EXCEEDED" /opt/alphapulse-rl/logs/

# Analyze trade patterns
awk -F',' '{print $3}' /opt/alphapulse-rl/logs/trades.csv | sort | uniq -c
```

## Emergency Procedures

### Emergency Stop Trading

**When to Use**: Critical system issues, extreme market conditions, regulatory requirements

**Procedure**:
```bash
# 1. Stop trading immediately
curl -X POST http://localhost:8080/emergency_stop

# 2. Flatten all positions
curl -X POST http://localhost:8080/flatten_positions

# 3. Stop trading system
docker-compose -f /opt/alphapulse-rl/deployment/docker/docker-compose.yml stop alphapulse-trading

# 4. Verify positions are closed
curl http://localhost:8080/metrics | jq '.portfolio.positions'

# 5. Document the incident
echo "$(date): Emergency stop initiated - [REASON]" >> /opt/alphapulse-rl/logs/incidents.log
```

### System Recovery

**After Emergency Stop**:

1. **Investigate Root Cause**
   ```bash
   # Review logs for errors
   grep -i error /opt/alphapulse-rl/logs/alphapulse_*.log | tail -100
   
   # Check system resources
   htop
   df -h
   
   # Verify data integrity
   python /opt/alphapulse-rl/data/weex_fetcher.py --validate
   ```

2. **Fix Issues**
   - Address identified problems
   - Update configuration if needed
   - Test fixes in paper trading mode

3. **Gradual Restart**
   ```bash
   # Start in paper trading mode
   sed -i 's/TRADING_MODE=live/TRADING_MODE=paper/' /opt/alphapulse-rl/deployment/.env
   
   # Restart system
   docker-compose -f /opt/alphapulse-rl/deployment/docker/docker-compose.yml up -d
   
   # Monitor for 30 minutes
   # If stable, switch back to live trading
   sed -i 's/TRADING_MODE=paper/TRADING_MODE=live/' /opt/alphapulse-rl/deployment/.env
   docker-compose restart alphapulse-trading
   ```

### Disaster Recovery

**Complete System Failure**:

1. **Assess Damage**
   ```bash
   # Check what's recoverable
   ls -la /opt/alphapulse-rl/
   
   # Check backup availability
   ls -la /opt/alphapulse-rl/backups/
   ```

2. **Restore from Backup**
   ```bash
   # Find latest backup
   python /opt/alphapulse-rl/deployment/scripts/backup_system.py list
   
   # Restore system
   python /opt/alphapulse-rl/deployment/scripts/backup_system.py restore --backup-name LATEST_BACKUP
   ```

3. **Verify Recovery**
   ```bash
   # Test system functionality
   ./deployment/scripts/deploy.sh
   
   # Verify data integrity
   python /opt/alphapulse-rl/models/evaluate.py --validate
   ```

## Maintenance Procedures

### Scheduled Maintenance

**Monthly Maintenance Window** (Recommended: First Sunday of month, 2-4 AM)

1. **Pre-Maintenance**
   ```bash
   # Create backup
   python /opt/alphapulse-rl/deployment/scripts/backup_system.py create --type full
   
   # Stop trading
   curl -X POST http://localhost:8080/maintenance_mode
   ```

2. **System Updates**
   ```bash
   # Update system packages
   sudo apt update && sudo apt upgrade -y
   
   # Update Docker images
   docker-compose -f /opt/alphapulse-rl/deployment/docker/docker-compose.yml pull
   
   # Update Python dependencies
   pip install -r requirements.txt --upgrade
   ```

3. **Database Maintenance**
   ```bash
   # Clean up old logs
   find /opt/alphapulse-rl/logs -name "*.log" -mtime +30 -delete
   
   # Optimize Redis
   redis-cli BGREWRITEAOF
   
   # Clean up temporary files
   find /opt/alphapulse-rl -name "*.tmp" -delete
   ```

4. **Post-Maintenance**
   ```bash
   # Restart services
   docker-compose -f /opt/alphapulse-rl/deployment/docker/docker-compose.yml restart
   
   # Verify functionality
   ./deployment/scripts/health_check.py
   
   # Resume trading
   curl -X POST http://localhost:8080/resume_trading
   ```

### Model Updates

**When to Update**:
- Weekly performance review shows degradation
- New market conditions require adaptation
- Significant changes in trading pairs or exchange

**Procedure**:
```bash
# 1. Train new model
python /opt/alphapulse-rl/models/train.py --config production

# 2. Validate new model
python /opt/alphapulse-rl/models/evaluate.py --model new_model.pth

# 3. Test in paper trading
sed -i 's/TRADING_MODE=live/TRADING_MODE=paper/' /opt/alphapulse-rl/deployment/.env
# Deploy new model and test for 24 hours

# 4. Deploy to live trading if successful
sed -i 's/TRADING_MODE=paper/TRADING_MODE=live/' /opt/alphapulse-rl/deployment/.env
docker-compose restart alphapulse-trading
```

## Performance Optimization

### System Performance Tuning

1. **Docker Resource Limits**
   ```yaml
   # In docker-compose.yml
   services:
     alphapulse-trading:
       deploy:
         resources:
           limits:
             cpus: '4.0'
             memory: 8G
           reservations:
             cpus: '2.0'
             memory: 4G
   ```

2. **Redis Optimization**
   ```bash
   # Optimize Redis configuration
   redis-cli CONFIG SET maxmemory 2gb
   redis-cli CONFIG SET maxmemory-policy allkeys-lru
   redis-cli CONFIG SET save "900 1 300 10 60 10000"
   ```

3. **System Tuning**
   ```bash
   # Increase file descriptor limits
   echo "alphapulse soft nofile 65536" | sudo tee -a /etc/security/limits.conf
   echo "alphapulse hard nofile 65536" | sudo tee -a /etc/security/limits.conf
   
   # Optimize network settings
   echo "net.core.rmem_max = 16777216" | sudo tee -a /etc/sysctl.conf
   echo "net.core.wmem_max = 16777216" | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p
   ```

### Application Performance

1. **Model Inference Optimization**
   ```python
   # Enable batch processing
   BATCH_PROCESSING_ENABLED=true
   
   # Optimize model loading
   CACHE_ENABLED=true
   MODEL_CACHE_SIZE=3
   ```

2. **Data Processing Optimization**
   ```python
   # Use vectorized operations
   VECTORIZED_FEATURES=true
   
   # Enable data caching
   DATA_CACHE_TTL=300
   ```

## Security Operations

### Access Control

1. **User Management**
   ```bash
   # Create monitoring user
   sudo useradd -m -s /bin/bash alphapulse-monitor
   sudo usermod -aG docker alphapulse-monitor
   
   # Set up SSH key authentication
   sudo mkdir -p /home/alphapulse-monitor/.ssh
   sudo cp authorized_keys /home/alphapulse-monitor/.ssh/
   sudo chown -R alphapulse-monitor:alphapulse-monitor /home/alphapulse-monitor/.ssh
   sudo chmod 700 /home/alphapulse-monitor/.ssh
   sudo chmod 600 /home/alphapulse-monitor/.ssh/authorized_keys
   ```

2. **API Security**
   ```bash
   # Rotate API keys monthly
   # 1. Generate new keys on exchange
   # 2. Update .env file
   # 3. Restart services
   # 4. Verify functionality
   # 5. Disable old keys
   ```

3. **Network Security**
   ```bash
   # Configure firewall
   sudo ufw allow from TRUSTED_IP to any port 22
   sudo ufw allow from TRUSTED_IP to any port 8080
   sudo ufw allow from TRUSTED_IP to any port 8081
   sudo ufw deny 22
   sudo ufw deny 8080
   sudo ufw deny 8081
   sudo ufw enable
   ```

### Security Monitoring

1. **Log Monitoring**
   ```bash
   # Monitor authentication attempts
   sudo tail -f /var/log/auth.log | grep alphapulse
   
   # Check for suspicious API activity
   grep -i "unauthorized\|forbidden\|failed" /opt/alphapulse-rl/logs/alphapulse_*.log
   ```

2. **Intrusion Detection**
   ```bash
   # Install and configure fail2ban
   sudo apt install fail2ban
   sudo systemctl enable fail2ban
   sudo systemctl start fail2ban
   ```

## Backup and Recovery

### Backup Strategy

1. **Automated Backups**
   ```bash
   # Daily incremental backups
   0 2 * * * /opt/alphapulse-rl/deployment/scripts/backup_system.py create --type data
   
   # Weekly full backups
   0 1 * * 0 /opt/alphapulse-rl/deployment/scripts/backup_system.py create --type full
   
   # Monthly configuration backups
   0 0 1 * * /opt/alphapulse-rl/deployment/scripts/backup_system.py create --type config
   ```

2. **Backup Verification**
   ```bash
   # Test backup integrity weekly
   python /opt/alphapulse-rl/deployment/scripts/backup_system.py list
   python /opt/alphapulse-rl/deployment/scripts/backup_system.py restore --backup-name TEST_BACKUP --dry-run
   ```

3. **Offsite Backup**
   ```bash
   # Configure S3 backup
   export BACKUP_S3_ENABLED=true
   export BACKUP_S3_BUCKET=alphapulse-backups
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   ```

### Recovery Procedures

1. **Partial Recovery** (Configuration only)
   ```bash
   python /opt/alphapulse-rl/deployment/scripts/backup_system.py restore --backup-name BACKUP_NAME --components config
   ```

2. **Full System Recovery**
   ```bash
   # Stop services
   docker-compose -f /opt/alphapulse-rl/deployment/docker/docker-compose.yml down
   
   # Restore from backup
   python /opt/alphapulse-rl/deployment/scripts/backup_system.py restore --backup-name BACKUP_NAME
   
   # Restart services
   docker-compose -f /opt/alphapulse-rl/deployment/docker/docker-compose.yml up -d
   
   # Verify recovery
   ./deployment/scripts/health_check.py
   ```

## Incident Response

### Incident Classification

- **P1 (Critical)**: Trading system down, major financial loss, security breach
- **P2 (High)**: Performance degradation, minor financial impact, partial outage
- **P3 (Medium)**: Non-critical issues, monitoring alerts, planned maintenance
- **P4 (Low)**: Documentation updates, minor improvements, informational

### Response Procedures

1. **Incident Detection**
   - Automated alerts via monitoring system
   - Manual detection during routine checks
   - External notification (exchange issues, etc.)

2. **Initial Response** (Within 5 minutes)
   ```bash
   # Document incident
   echo "$(date): INCIDENT DETECTED - [DESCRIPTION]" >> /opt/alphapulse-rl/logs/incidents.log
   
   # Assess severity
   # P1: Immediate emergency stop if needed
   # P2: Begin investigation
   # P3/P4: Schedule resolution
   ```

3. **Investigation and Resolution**
   - Gather relevant logs and metrics
   - Identify root cause
   - Implement fix or workaround
   - Test resolution
   - Document lessons learned

4. **Post-Incident Review**
   - Analyze incident timeline
   - Identify improvement opportunities
   - Update procedures if needed
   - Communicate findings to stakeholders

### Contact Information

**Emergency Contacts**:
- System Administrator: [PHONE] / [EMAIL]
- Trading Team Lead: [PHONE] / [EMAIL]
- Exchange Support: [PHONE] / [EMAIL]
- Infrastructure Team: [PHONE] / [EMAIL]

**Escalation Matrix**:
- P1 Incidents: Immediate notification to all contacts
- P2 Incidents: Notify within 30 minutes
- P3 Incidents: Notify within 4 hours
- P4 Incidents: Include in daily summary

---

## Appendix

### Useful Commands Reference

```bash
# System Status
docker-compose -f /opt/alphapulse-rl/deployment/docker/docker-compose.yml ps
sudo systemctl status alphapulse-trading
curl http://localhost:8080/health
curl http://localhost:8081/health

# Logs
tail -f /opt/alphapulse-rl/logs/alphapulse_$(date +%Y%m%d).log
docker logs alphapulse-trading
journalctl -u alphapulse-trading -f

# Performance
htop
df -h
free -h
docker stats

# Trading Operations
curl http://localhost:8080/metrics
curl -X POST http://localhost:8080/emergency_stop
curl -X POST http://localhost:8080/flatten_positions

# Backup Operations
python /opt/alphapulse-rl/deployment/scripts/backup_system.py create
python /opt/alphapulse-rl/deployment/scripts/backup_system.py list
python /opt/alphapulse-rl/deployment/scripts/backup_system.py restore --backup-name BACKUP_NAME
```

### Configuration Files

- **Main Config**: `/opt/alphapulse-rl/config/config.yaml`
- **Trading Params**: `/opt/alphapulse-rl/config/trading_params.yaml`
- **Environment**: `/opt/alphapulse-rl/deployment/.env`
- **Docker Compose**: `/opt/alphapulse-rl/deployment/docker/docker-compose.yml`
- **Nginx Config**: `/opt/alphapulse-rl/deployment/docker/nginx.conf`

### Performance Baselines

- **CPU Usage**: Normal < 50%, Warning > 80%, Critical > 90%
- **Memory Usage**: Normal < 60%, Warning > 80%, Critical > 90%
- **Disk Usage**: Normal < 70%, Warning > 80%, Critical > 90%
- **API Response Time**: Normal < 500ms, Warning > 1s, Critical > 5s
- **Model Inference**: Normal < 100ms, Warning > 500ms, Critical > 1s

This operations manual should be reviewed and updated quarterly to ensure accuracy and completeness.
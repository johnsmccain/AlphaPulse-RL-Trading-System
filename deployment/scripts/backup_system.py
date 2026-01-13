#!/usr/bin/env python3
"""
AlphaPulse-RL System Backup Script

Automated backup system for configuration, data, models, and logs with
support for local and cloud storage.
"""

import os
import sys
import json
import shutil
import tarfile
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackupManager:
    """Manages system backups with configurable retention and storage options."""
    
    def __init__(self):
        self.backup_root = os.getenv('BACKUP_ROOT', 'backups')
        self.retention_days = int(os.getenv('BACKUP_RETENTION_DAYS', '30'))
        self.enable_compression = os.getenv('BACKUP_COMPRESSION', 'true').lower() == 'true'
        self.enable_cloud_backup = os.getenv('BACKUP_S3_ENABLED', 'false').lower() == 'true'
        self.s3_bucket = os.getenv('BACKUP_S3_BUCKET', '')
        
        # Ensure backup directory exists
        os.makedirs(self.backup_root, exist_ok=True)
    
    def create_backup(self, backup_type: str = 'full') -> str:
        """Create a system backup."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"alphapulse_backup_{backup_type}_{timestamp}"
        backup_dir = os.path.join(self.backup_root, backup_name)
        
        logger.info(f"Creating {backup_type} backup: {backup_name}")
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup manifest
            manifest = {
                'backup_name': backup_name,
                'backup_type': backup_type,
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'components': []
            }
            
            # Backup components based on type
            if backup_type in ['full', 'config']:
                self._backup_configuration(backup_dir, manifest)
            
            if backup_type in ['full', 'data']:
                self._backup_data(backup_dir, manifest)
            
            if backup_type in ['full', 'models']:
                self._backup_models(backup_dir, manifest)
            
            if backup_type in ['full', 'logs']:
                self._backup_logs(backup_dir, manifest)
            
            # Save manifest
            manifest_file = os.path.join(backup_dir, 'backup_manifest.json')
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Compress backup if enabled
            if self.enable_compression:
                compressed_backup = self._compress_backup(backup_dir)
                shutil.rmtree(backup_dir)  # Remove uncompressed version
                backup_path = compressed_backup
            else:
                backup_path = backup_dir
            
            # Upload to cloud if enabled
            if self.enable_cloud_backup and self.s3_bucket:
                self._upload_to_s3(backup_path)
            
            logger.info(f"Backup completed successfully: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            # Cleanup failed backup
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            raise
    
    def _backup_configuration(self, backup_dir: str, manifest: Dict[str, Any]):
        """Backup configuration files."""
        logger.info("Backing up configuration files...")
        
        config_backup_dir = os.path.join(backup_dir, 'config')
        os.makedirs(config_backup_dir, exist_ok=True)
        
        config_files = [
            'config/config.yaml',
            'config/trading_params.yaml',
            'config/logging_config.py',
            'deployment/.env',
            'deployment/docker/docker-compose.yml',
            'deployment/docker/nginx.conf',
            'deployment/monitoring/requirements.txt'
        ]
        
        backed_up_files = []
        
        for config_file in config_files:
            if os.path.exists(config_file):
                dest_path = os.path.join(config_backup_dir, os.path.basename(config_file))
                shutil.copy2(config_file, dest_path)
                backed_up_files.append(config_file)
                logger.debug(f"Backed up: {config_file}")
        
        manifest['components'].append({
            'type': 'configuration',
            'files_count': len(backed_up_files),
            'files': backed_up_files
        })
        
        logger.info(f"Configuration backup completed: {len(backed_up_files)} files")
    
    def _backup_data(self, backup_dir: str, manifest: Dict[str, Any]):
        """Backup data files."""
        logger.info("Backing up data files...")
        
        data_backup_dir = os.path.join(backup_dir, 'data')
        
        if os.path.exists('data'):
            shutil.copytree('data', data_backup_dir, ignore=shutil.ignore_patterns('*.tmp', '__pycache__'))
            
            # Count files
            file_count = sum([len(files) for r, d, files in os.walk(data_backup_dir)])
            
            manifest['components'].append({
                'type': 'data',
                'files_count': file_count,
                'size_mb': self._get_directory_size(data_backup_dir) / (1024 * 1024)
            })
            
            logger.info(f"Data backup completed: {file_count} files")
        else:
            logger.warning("Data directory not found, skipping data backup")
    
    def _backup_models(self, backup_dir: str, manifest: Dict[str, Any]):
        """Backup model files."""
        logger.info("Backing up model files...")
        
        models_backup_dir = os.path.join(backup_dir, 'models')
        
        if os.path.exists('models'):
            # Only backup saved models, not temporary files
            os.makedirs(models_backup_dir, exist_ok=True)
            
            model_files = []
            for root, dirs, files in os.walk('models'):
                for file in files:
                    if file.endswith(('.pth', '.pkl', '.joblib', '.h5', '.pb')):
                        src_path = os.path.join(root, file)
                        rel_path = os.path.relpath(src_path, 'models')
                        dest_path = os.path.join(models_backup_dir, rel_path)
                        
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(src_path, dest_path)
                        model_files.append(src_path)
            
            manifest['components'].append({
                'type': 'models',
                'files_count': len(model_files),
                'files': model_files,
                'size_mb': self._get_directory_size(models_backup_dir) / (1024 * 1024)
            })
            
            logger.info(f"Models backup completed: {len(model_files)} files")
        else:
            logger.warning("Models directory not found, skipping models backup")
    
    def _backup_logs(self, backup_dir: str, manifest: Dict[str, Any]):
        """Backup recent log files."""
        logger.info("Backing up log files...")
        
        logs_backup_dir = os.path.join(backup_dir, 'logs')
        
        if os.path.exists('logs'):
            os.makedirs(logs_backup_dir, exist_ok=True)
            
            # Only backup recent logs (last 7 days)
            cutoff_date = datetime.now() - timedelta(days=7)
            backed_up_logs = []
            
            for filename in os.listdir('logs'):
                filepath = os.path.join('logs', filename)
                
                if os.path.isfile(filepath):
                    # Check file modification time
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if mtime > cutoff_date:
                        dest_path = os.path.join(logs_backup_dir, filename)
                        shutil.copy2(filepath, dest_path)
                        backed_up_logs.append(filename)
            
            manifest['components'].append({
                'type': 'logs',
                'files_count': len(backed_up_logs),
                'files': backed_up_logs,
                'size_mb': self._get_directory_size(logs_backup_dir) / (1024 * 1024)
            })
            
            logger.info(f"Logs backup completed: {len(backed_up_logs)} files")
        else:
            logger.warning("Logs directory not found, skipping logs backup")
    
    def _compress_backup(self, backup_dir: str) -> str:
        """Compress backup directory."""
        logger.info("Compressing backup...")
        
        compressed_file = f"{backup_dir}.tar.gz"
        
        with tarfile.open(compressed_file, 'w:gz') as tar:
            tar.add(backup_dir, arcname=os.path.basename(backup_dir))
        
        # Get compression ratio
        original_size = self._get_directory_size(backup_dir)
        compressed_size = os.path.getsize(compressed_file)
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        logger.info(f"Compression completed: {compression_ratio:.1f}% reduction")
        
        return compressed_file
    
    def _upload_to_s3(self, backup_path: str):
        """Upload backup to S3."""
        if not self.s3_bucket:
            logger.warning("S3 bucket not configured, skipping cloud upload")
            return
        
        logger.info(f"Uploading backup to S3: {self.s3_bucket}")
        
        try:
            # Use AWS CLI for upload
            backup_filename = os.path.basename(backup_path)
            s3_key = f"alphapulse-backups/{backup_filename}"
            
            cmd = [
                'aws', 's3', 'cp', backup_path, f"s3://{self.s3_bucket}/{s3_key}",
                '--storage-class', 'STANDARD_IA'  # Use cheaper storage class
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info("S3 upload completed successfully")
            else:
                logger.error(f"S3 upload failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("S3 upload timed out")
        except Exception as e:
            logger.error(f"S3 upload error: {str(e)}")
    
    def _get_directory_size(self, directory: str) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    def cleanup_old_backups(self):
        """Remove old backups based on retention policy."""
        logger.info(f"Cleaning up backups older than {self.retention_days} days")
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        removed_count = 0
        
        for item in os.listdir(self.backup_root):
            item_path = os.path.join(self.backup_root, item)
            
            if os.path.isfile(item_path) or os.path.isdir(item_path):
                # Check modification time
                mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                
                if mtime < cutoff_date:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    else:
                        shutil.rmtree(item_path)
                    
                    logger.info(f"Removed old backup: {item}")
                    removed_count += 1
        
        logger.info(f"Cleanup completed: {removed_count} old backups removed")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        
        for item in os.listdir(self.backup_root):
            item_path = os.path.join(self.backup_root, item)
            
            if os.path.exists(item_path):
                # Try to read manifest
                manifest_file = None
                
                if os.path.isdir(item_path):
                    manifest_file = os.path.join(item_path, 'backup_manifest.json')
                elif item.endswith('.tar.gz'):
                    # Extract manifest from compressed backup
                    try:
                        with tarfile.open(item_path, 'r:gz') as tar:
                            manifest_member = None
                            for member in tar.getmembers():
                                if member.name.endswith('backup_manifest.json'):
                                    manifest_member = member
                                    break
                            
                            if manifest_member:
                                manifest_content = tar.extractfile(manifest_member).read()
                                manifest_data = json.loads(manifest_content.decode('utf-8'))
                                
                                backup_info = {
                                    'name': item,
                                    'path': item_path,
                                    'size_mb': os.path.getsize(item_path) / (1024 * 1024),
                                    'created_at': manifest_data.get('created_at'),
                                    'backup_type': manifest_data.get('backup_type'),
                                    'components': len(manifest_data.get('components', []))
                                }
                                backups.append(backup_info)
                    except Exception as e:
                        logger.warning(f"Could not read manifest from {item}: {str(e)}")
                
                elif os.path.exists(manifest_file):
                    try:
                        with open(manifest_file, 'r') as f:
                            manifest_data = json.load(f)
                        
                        backup_info = {
                            'name': item,
                            'path': item_path,
                            'size_mb': self._get_directory_size(item_path) / (1024 * 1024),
                            'created_at': manifest_data.get('created_at'),
                            'backup_type': manifest_data.get('backup_type'),
                            'components': len(manifest_data.get('components', []))
                        }
                        backups.append(backup_info)
                    except Exception as e:
                        logger.warning(f"Could not read manifest from {manifest_file}: {str(e)}")
        
        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return backups
    
    def restore_backup(self, backup_name: str, components: List[str] = None):
        """Restore from backup."""
        logger.info(f"Restoring from backup: {backup_name}")
        
        backup_path = os.path.join(self.backup_root, backup_name)
        
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # Extract if compressed
        if backup_name.endswith('.tar.gz'):
            logger.info("Extracting compressed backup...")
            extract_dir = backup_path.replace('.tar.gz', '_extracted')
            
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(extract_dir))
            
            backup_path = extract_dir
        
        # Read manifest
        manifest_file = os.path.join(backup_path, 'backup_manifest.json')
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Backup manifest not found: {manifest_file}")
        
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        # Restore components
        if components is None:
            components = [comp['type'] for comp in manifest['components']]
        
        for component in components:
            self._restore_component(backup_path, component)
        
        logger.info("Restore completed successfully")
    
    def _restore_component(self, backup_path: str, component: str):
        """Restore a specific component."""
        logger.info(f"Restoring component: {component}")
        
        component_path = os.path.join(backup_path, component)
        
        if not os.path.exists(component_path):
            logger.warning(f"Component not found in backup: {component}")
            return
        
        if component == 'config':
            # Restore configuration files
            for filename in os.listdir(component_path):
                src_file = os.path.join(component_path, filename)
                
                # Determine destination based on filename
                if filename == '.env':
                    dest_file = 'deployment/.env'
                elif filename in ['config.yaml', 'trading_params.yaml']:
                    dest_file = f'config/{filename}'
                else:
                    dest_file = filename
                
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                shutil.copy2(src_file, dest_file)
                logger.debug(f"Restored: {dest_file}")
        
        elif component in ['data', 'models', 'logs']:
            # Restore directory
            if os.path.exists(component):
                # Backup existing directory
                backup_existing = f"{component}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.move(component, backup_existing)
                logger.info(f"Existing {component} backed up to: {backup_existing}")
            
            shutil.copytree(component_path, component)
            logger.info(f"Restored {component} directory")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaPulse-RL Backup Manager')
    parser.add_argument('action', choices=['create', 'list', 'restore', 'cleanup'],
                       help='Action to perform')
    parser.add_argument('--type', choices=['full', 'config', 'data', 'models', 'logs'],
                       default='full', help='Backup type (for create action)')
    parser.add_argument('--backup-name', help='Backup name (for restore action)')
    parser.add_argument('--components', nargs='+', 
                       choices=['config', 'data', 'models', 'logs'],
                       help='Components to restore (for restore action)')
    
    args = parser.parse_args()
    
    backup_manager = BackupManager()
    
    try:
        if args.action == 'create':
            backup_path = backup_manager.create_backup(args.type)
            print(f"Backup created: {backup_path}")
        
        elif args.action == 'list':
            backups = backup_manager.list_backups()
            
            if not backups:
                print("No backups found")
            else:
                print(f"{'Name':<40} {'Type':<10} {'Size (MB)':<10} {'Created':<20}")
                print("-" * 80)
                
                for backup in backups:
                    print(f"{backup['name']:<40} {backup.get('backup_type', 'unknown'):<10} "
                          f"{backup['size_mb']:<10.1f} {backup.get('created_at', 'unknown'):<20}")
        
        elif args.action == 'restore':
            if not args.backup_name:
                print("Error: --backup-name is required for restore action")
                sys.exit(1)
            
            backup_manager.restore_backup(args.backup_name, args.components)
            print("Restore completed successfully")
        
        elif args.action == 'cleanup':
            backup_manager.cleanup_old_backups()
            print("Cleanup completed")
    
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
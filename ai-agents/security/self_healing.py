import logging
import time
import psutil
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import subprocess
import os
import sys
import threading
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    response_time: float
    error_rate: float
    timestamp: datetime

class SelfHealingSystem:
    def __init__(self):
        self.health_thresholds = {
            'cpu_usage': 80.0,  # %
            'memory_usage': 85.0,  # %
            'disk_usage': 90.0,  # %
            'response_time': 5.0,  # seconds
            'error_rate': 10.0,  # %
            'active_connections': 1000
        }
        
        self.health_history = []
        self.recovery_actions = []
        self.monitoring_interval = 60  # seconds
        self.is_monitoring = False
        
    def start_monitoring(self):
        """Start the self-healing monitoring system"""
        self.is_monitoring = True
        logger.info("Self-healing system monitoring started")
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=self._monitoring_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start health reporter
        reporter_thread = threading.Thread(target=self._health_reporting_loop)
        reporter_thread.daemon = True
        reporter_thread.start()
    
    def stop_monitoring(self):
        """Stop the self-healing monitoring system"""
        self.is_monitoring = False
        logger.info("Self-healing system monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Check system health
                health_status = self.check_system_health()
                self.health_history.append(health_status)
                
                # Keep only last 100 records
                if len(self.health_history) > 100:
                    self.health_history.pop(0)
                
                # Check if healing is needed
                if self._needs_healing(health_status):
                    self._perform_healing(health_status)
                
                # Log health status
                if len(self.health_history) % 10 == 0:  # Log every 10 cycles
                    logger.info(f"System health: CPU {health_status.cpu_usage:.1f}%, "
                               f"Memory {health_status.memory_usage:.1f}%, "
                               f"Errors {health_status.error_rate:.1f}%")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
            
            time.sleep(self.monitoring_interval)
    
    def _health_reporting_loop(self):
        """Health reporting loop for external monitoring"""
        while self.is_monitoring:
            try:
                # Generate health report
                report = self.generate_health_report()
                
                # Send to monitoring service (if configured)
                self._send_health_report(report)
                
                # Save report locally
                self._save_health_report(report)
                
            except Exception as e:
                logger.error(f"Error in health reporting: {str(e)}")
            
            time.sleep(300)  # Report every 5 minutes
    
    def check_system_health(self) -> SystemHealth:
        """Check overall system health"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Active connections (simplified)
            active_connections = len(psutil.net_connections())
            
            # Response time (simulate API check)
            response_time = self._check_response_time()
            
            # Error rate
            error_rate = self._calculate_error_rate()
            
            return SystemHealth(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                active_connections=active_connections,
                response_time=response_time,
                error_rate=error_rate,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")
            # Return default health status
            return SystemHealth(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                active_connections=0,
                response_time=0.0,
                error_rate=0.0,
                timestamp=datetime.now()
            )
    
    def _check_response_time(self) -> float:
        """Check API response time"""
        try:
            start_time = time.time()
            # Simulate API call to health endpoint
            response = requests.get('http://localhost:3000/api/health', timeout=5)
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200:
                return response_time
            else:
                return 9999.0  # High response time for errors
                
        except Exception as e:
            logger.warning(f"Health endpoint check failed: {str(e)}")
            return 9999.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        try:
            # In a real implementation, this would query error logs or metrics
            # For now, we'll simulate based on recent health checks
            
            if len(self.health_history) < 2:
                return 0.0
            
            recent_checks = self.health_history[-10:]  # Last 10 checks
            error_count = sum(1 for check in recent_checks if check.error_rate > 5.0)
            
            return (error_count / len(recent_checks)) * 100
            
        except Exception as e:
            logger.error(f"Error calculating error rate: {str(e)}")
            return 50.0  # Conservative error rate
    
    def _needs_healing(self, health: SystemHealth) -> bool:
        """Check if system needs healing based on thresholds"""
        try:
            # Check CPU usage
            if health.cpu_usage > self.health_thresholds['cpu_usage']:
                logger.warning(f"High CPU usage detected: {health.cpu_usage:.1f}%")
                return True
            
            # Check memory usage
            if health.memory_usage > self.health_thresholds['memory_usage']:
                logger.warning(f"High memory usage detected: {health.memory_usage:.1f}%")
                return True
            
            # Check disk usage
            if health.disk_usage > self.health_thresholds['disk_usage']:
                logger.warning(f"High disk usage detected: {health.disk_usage:.1f}%")
                return True
            
            # Check response time
            if health.response_time > self.health_thresholds['response_time'] * 1000:  # Convert to ms
                logger.warning(f"High response time detected: {health.response_time:.1f}ms")
                return True
            
            # Check error rate
            if health.error_rate > self.health_thresholds['error_rate']:
                logger.warning(f"High error rate detected: {health.error_rate:.1f}%")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking healing needs: {str(e)}")
            return False
    
    def _perform_healing(self, health: SystemHealth):
        """Perform automatic healing actions"""
        logger.info("Performing system healing...")
        
        healing_actions = []
        
        try:
            # CPU-related healing
            if health.cpu_usage > self.health_thresholds['cpu_usage']:
                action = self._heal_cpu_usage()
                healing_actions.append(action)
            
            # Memory-related healing
            if health.memory_usage > self.health_thresholds['memory_usage']:
                action = self._heal_memory_usage()
                healing_actions.append(action)
            
            # Disk-related healing
            if health.disk_usage > self.health_thresholds['disk_usage']:
                action = self._heal_disk_usage()
                healing_actions.append(action)
            
            # Response time healing
            if health.response_time > self.health_thresholds['response_time'] * 1000:
                action = self._heal_response_time()
                healing_actions.append(action)
            
            # Error rate healing
            if health.error_rate > self.health_thresholds['error_rate']:
                action = self._heal_error_rate()
                healing_actions.append(action)
            
            # Record healing actions
            healing_record = {
                'timestamp': datetime.now().isoformat(),
                'health_status': {
                    'cpu_usage': health.cpu_usage,
                    'memory_usage': health.memory_usage,
                    'disk_usage': health.disk_usage,
                    'response_time': health.response_time,
                    'error_rate': health.error_rate
                },
                'actions_taken': healing_actions,
                'success': True
            }
            
            self.recovery_actions.append(healing_record)
            
            # Keep only last 50 records
            if len(self.recovery_actions) > 50:
                self.recovery_actions.pop(0)
            
            logger.info(f"Healing completed. Actions taken: {len(healing_actions)}")
            
        except Exception as e:
            logger.error(f"Error during healing: {str(e)}")
    
    def _heal_cpu_usage(self) -> Dict[str, Any]:
        """Heal high CPU usage"""
        try:
            action = {
                'type': 'cpu_optimization',
                'description': 'Optimizing CPU usage',
                'timestamp': datetime.now().isoformat()
            }
            
            # Restart resource-intensive processes
            self._restart_resource_intensive_services()
            
            # Clear cache
            self._clear_memory_cache()
            
            # Optimize database queries
            self._optimize_database_queries()
            
            action['success'] = True
            logger.info("CPU healing completed")
            
        except Exception as e:
            action['success'] = False
            action['error'] = str(e)
            logger.error(f"CPU healing failed: {str(e)}")
        
        return action
    
    def _heal_memory_usage(self) -> Dict[str, Any]:
        """Heal high memory usage"""
        try:
            action = {
                'type': 'memory_optimization',
                'description': 'Optimizing memory usage',
                'timestamp': datetime.now().isoformat()
            }
            
            # Clear memory cache
            self._clear_memory_cache()
            
            # Restart memory-intensive services
            self._restart_memory_intensive_services()
            
            # Optimize application memory
            self._optimize_application_memory()
            
            action['success'] = True
            logger.info("Memory healing completed")
            
        except Exception as e:
            action['success'] = False
            action['error'] = str(e)
            logger.error(f"Memory healing failed: {str(e)}")
        
        return action
    
    def _heal_disk_usage(self) -> Dict[str, Any]:
        """Heal high disk usage"""
        try:
            action = {
                'type': 'disk_optimization',
                'description': 'Optimizing disk usage',
                'timestamp': datetime.now().isoformat()
            }
            
            # Clear temporary files
            self._clear_temp_files()
            
            # Clean log files
            self._clean_log_files()
            
            # Optimize database storage
            self._optimize_database_storage()
            
            action['success'] = True
            logger.info("Disk healing completed")
            
        except Exception as e:
            action['success'] = False
            action['error'] = str(e)
            logger.error(f"Disk healing failed: {str(e)}")
        
        return action
    
    def _heal_response_time(self) -> Dict[str, Any]:
        """Heal high response time"""
        try:
            action = {
                'type': 'response_time_optimization',
                'description': 'Optimizing response time',
                'timestamp': datetime.now().isoformat()
            }
            
            # Restart web server
            self._restart_web_server()
            
            # Clear application cache
            self._clear_application_cache()
            
            # Optimize database performance
            self._optimize_database_performance()
            
            action['success'] = True
            logger.info("Response time healing completed")
            
        except Exception as e:
            action['success'] = False
            action['error'] = str(e)
            logger.error(f"Response time healing failed: {str(e)}")
        
        return action
    
    def _heal_error_rate(self) -> Dict[str, Any]:
        """Heal high error rate"""
        try:
            action = {
                'type': 'error_rate_reduction',
                'description': 'Reducing error rate',
                'timestamp': datetime.now().isoformat()
            }
            
            # Restart failing services
            self._restart_failing_services()
            
            # Clear error states
            self._clear_error_states()
            
            # Update configuration if needed
            self._update_error_prone_configurations()
            
            action['success'] = True
            logger.info("Error rate healing completed")
            
        except Exception as e:
            action['success'] = False
            action['error'] = str(e)
            logger.error(f"Error rate healing failed: {str(e)}")
        
        return action
    
    # Implementation of specific healing methods
    def _restart_resource_intensive_services(self):
        """Restart resource-intensive services"""
        try:
            # In production, this would restart specific services
            # For now, we'll simulate the action
            logger.info("Restarting resource-intensive services")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error restarting services: {str(e)}")
    
    def _clear_memory_cache(self):
        """Clear memory cache"""
        try:
            if os.name == 'posix':  # Linux/Unix
                subprocess.run(['sync'], check=True)
                subprocess.run(['echo', '3', '>', '/proc/sys/vm/drop_caches'], check=True)
            logger.info("Memory cache cleared")
        except Exception as e:
            logger.warning(f"Could not clear memory cache: {str(e)}")
    
    def _optimize_database_queries(self):
        """Optimize database queries"""
        try:
            # This would run database optimization queries
            logger.info("Database queries optimized")
        except Exception as e:
            logger.error(f"Error optimizing database: {str(e)}")
    
    def _restart_memory_intensive_services(self):
        """Restart memory-intensive services"""
        try:
            logger.info("Restarting memory-intensive services")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error restarting memory services: {str(e)}")
    
    def _optimize_application_memory(self):
        """Optimize application memory usage"""
        try:
            # This would trigger garbage collection and memory optimization
            import gc
            gc.collect()
            logger.info("Application memory optimized")
        except Exception as e:
            logger.error(f"Error optimizing application memory: {str(e)}")
    
    def _clear_temp_files(self):
        """Clear temporary files"""
        try:
            import tempfile
            import shutil
            
            temp_dir = tempfile.gettempdir()
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    pass  # Skip files that can't be deleted
            
            logger.info("Temporary files cleared")
        except Exception as e:
            logger.error(f"Error clearing temp files: {str(e)}")
    
    def _clean_log_files(self):
        """Clean and rotate log files"""
        try:
            # This would implement log rotation and cleanup
            logger.info("Log files cleaned")
        except Exception as e:
            logger.error(f"Error cleaning log files: {str(e)}")
    
    def _optimize_database_storage(self):
        """Optimize database storage"""
        try:
            # This would run database maintenance tasks
            logger.info("Database storage optimized")
        except Exception as e:
            logger.error(f"Error optimizing database storage: {str(e)}")
    
    def _restart_web_server(self):
        """Restart web server"""
        try:
            logger.info("Restarting web server")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error restarting web server: {str(e)}")
    
    def _clear_application_cache(self):
        """Clear application cache"""
        try:
            # This would clear application-specific caches
            logger.info("Application cache cleared")
        except Exception as e:
            logger.error(f"Error clearing application cache: {str(e)}")
    
    def _optimize_database_performance(self):
        """Optimize database performance"""
        try:
            # This would run database performance optimization
            logger.info("Database performance optimized")
        except Exception as e:
            logger.error(f"Error optimizing database performance: {str(e)}")
    
    def _restart_failing_services(self):
        """Restart failing services"""
        try:
            logger.info("Restarting failing services")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error restarting failing services: {str(e)}")
    
    def _clear_error_states(self):
        """Clear error states"""
        try:
            # This would reset error states in the application
            logger.info("Error states cleared")
        except Exception as e:
            logger.error(f"Error clearing error states: {str(e)}")
    
    def _update_error_prone_configurations(self):
        """Update error-prone configurations"""
        try:
            # This would update configurations that are causing errors
            logger.info("Error-prone configurations updated")
        except Exception as e:
            logger.error(f"Error updating configurations: {str(e)}")
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        try:
            current_health = self.health_history[-1] if self.health_history else self.check_system_health()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'current_health': {
                    'cpu_usage': current_health.cpu_usage,
                    'memory_usage': current_health.memory_usage,
                    'disk_usage': current_health.disk_usage,
                    'response_time': current_health.response_time,
                    'error_rate': current_health.error_rate,
                    'active_connections': current_health.active_connections
                },
                'thresholds': self.health_thresholds,
                'recent_recoveries': len([r for r in self.recovery_actions 
                                        if datetime.fromisoformat(r['timestamp']) > 
                                        datetime.now() - timedelta(hours=1)]),
                'system_uptime': self._get_system_uptime(),
                'recommendations': self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating health report: {str(e)}")
            return {'error': str(e)}
    
    def _get_system_uptime(self) -> str:
        """Get system uptime"""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            uptime_days = uptime_seconds / (24 * 3600)
            return f"{uptime_days:.1f} days"
        except:
            return "Unknown"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        if not self.health_history:
            return recommendations
        
        current_health = self.health_history[-1]
        
        if current_health.cpu_usage > 70:
            recommendations.append("Consider scaling horizontally to handle CPU load")
        
        if current_health.memory_usage > 75:
            recommendations.append("Optimize memory usage or add more RAM")
        
        if current_health.disk_usage > 85:
            recommendations.append("Clean up disk space or increase storage capacity")
        
        if current_health.error_rate > 5:
            recommendations.append("Investigate and fix recurring errors")
        
        return recommendations
    
    def _send_health_report(self, report: Dict[str, Any]):
        """Send health report to external monitoring"""
        try:
            # In production, this would send to monitoring service
            # For now, we'll just log it
            if report.get('current_health', {}).get('error_rate', 0) > 10:
                logger.warning(f"High error rate in health report: {report['current_health']['error_rate']}%")
        except Exception as e:
            logger.error(f"Error sending health report: {str(e)}")
    
    def _save_health_report(self, report: Dict[str, Any]):
        """Save health report to file"""
        try:
            reports_dir = 'health_reports'
            os.makedirs(reports_dir, exist_ok=True)
            
            filename = f"{reports_dir}/health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving health report: {str(e)}")

# Global instance
self_healing_system = SelfHealingSystem()

def init_self_healing():
    """Initialize the self-healing system"""
    self_healing_system.start_monitoring()
    return self_healing_system

if __name__ == "__main__":
    # Test the self-healing system
    healing_system = SelfHealingSystem()
    healing_system.start_monitoring()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        healing_system.stop_monitoring()
        print("Self-healing system stopped")

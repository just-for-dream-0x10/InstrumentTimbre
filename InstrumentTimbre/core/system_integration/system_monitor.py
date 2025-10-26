"""
System Monitor

Monitors system performance, resource usage, and health metrics for InstrumentTimbre.
"""

import psutil
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import json
import os

from .exception_types import SystemResourceError, InsufficientMemoryError, GPUNotAvailableError


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    temperature: Optional[float] = None
    network_io_mb: Optional[float] = None


@dataclass
class ProcessMetrics:
    """Process-specific metrics"""
    timestamp: float
    pid: int
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    threads: int
    open_files: int
    network_connections: int


@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    condition: Callable[[SystemMetrics], bool]
    severity: str
    message: str
    cooldown_seconds: float = 300.0
    enabled: bool = True
    last_triggered: Optional[float] = None


class SystemMonitor:
    """
    Comprehensive system monitoring for performance tracking,
    resource management, and proactive issue detection.
    """
    
    def __init__(self, monitoring_interval: float = 5.0, history_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Metrics storage
        self.system_metrics_history: deque = deque(maxlen=history_size)
        self.process_metrics_history: deque = deque(maxlen=history_size)
        self.alerts_history: List[Dict[str, Any]] = []
        
        # Current process info
        self.process = psutil.Process()
        
        # Alert system
        self.alert_rules: List[AlertRule] = []
        self.alert_callbacks: List[Callable] = []
        
        # Performance tracking
        self.performance_baselines = {
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "disk_threshold": 90.0,
            "response_time_threshold": 5.0
        }
        
        # Initialize alert rules
        self._initialize_default_alerts()
        
        # Check for GPU availability
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except ImportError:
            self.logger.info("GPUtil not available, GPU monitoring disabled")
            return False
        except Exception as e:
            self.logger.warning(f"GPU check failed: {e}")
            return False
    
    def _initialize_default_alerts(self):
        """Initialize default system alert rules"""
        
        default_alerts = [
            AlertRule(
                name="high_cpu_usage",
                condition=lambda m: m.cpu_percent > 90.0,
                severity="warning",
                message="High CPU usage detected: {cpu_percent:.1f}%",
                cooldown_seconds=300.0
            ),
            
            AlertRule(
                name="high_memory_usage",
                condition=lambda m: m.memory_percent > 90.0,
                severity="warning", 
                message="High memory usage detected: {memory_percent:.1f}%",
                cooldown_seconds=300.0
            ),
            
            AlertRule(
                name="low_memory_available",
                condition=lambda m: m.memory_available_mb < 500,
                severity="critical",
                message="Low memory available: {memory_available_mb:.0f}MB",
                cooldown_seconds=120.0
            ),
            
            AlertRule(
                name="high_disk_usage",
                condition=lambda m: m.disk_usage_percent > 95.0,
                severity="warning",
                message="High disk usage detected: {disk_usage_percent:.1f}%",
                cooldown_seconds=600.0
            ),
            
            AlertRule(
                name="gpu_memory_exhausted",
                condition=lambda m: m.gpu_memory_percent is not None and m.gpu_memory_percent > 95.0,
                severity="critical",
                message="GPU memory nearly exhausted: {gpu_memory_percent:.1f}%",
                cooldown_seconds=180.0
            )
        ]
        
        self.alert_rules.extend(default_alerts)
    
    def start_monitoring(self):
        """Start system monitoring"""
        
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="SystemMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info(f"System monitoring started (interval: {self.monitoring_interval}s)")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Collect process metrics
                process_metrics = self._collect_process_metrics()
                self.process_metrics_history.append(process_metrics)
                
                # Check alerts
                self._check_alerts(system_metrics)
                
                # Sleep until next interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        
        # Basic system metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics if available
        gpu_usage = None
        gpu_memory = None
        
        if self.gpu_available:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_usage = gpu.load * 100
                    gpu_memory = gpu.memoryUtil * 100
            except Exception as e:
                self.logger.debug(f"GPU metrics collection failed: {e}")
        
        # Network I/O
        network_io = None
        try:
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                bytes_sent_diff = net_io.bytes_sent - self._last_net_io.bytes_sent
                bytes_recv_diff = net_io.bytes_recv - self._last_net_io.bytes_recv
                network_io = (bytes_sent_diff + bytes_recv_diff) / 1024 / 1024  # MB
            self._last_net_io = net_io
        except Exception as e:
            self.logger.debug(f"Network I/O collection failed: {e}")
        
        # Temperature (if available)
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature
                for name, entries in temps.items():
                    if 'cpu' in name.lower() or 'core' in name.lower():
                        temperature = entries[0].current
                        break
        except Exception:
            pass  # Temperature monitoring not available on all systems
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_usage_percent=disk.percent,
            gpu_usage_percent=gpu_usage,
            gpu_memory_percent=gpu_memory,
            temperature=temperature,
            network_io_mb=network_io
        )
    
    def _collect_process_metrics(self) -> ProcessMetrics:
        """Collect current process metrics"""
        
        try:
            # Get process info
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            threads = self.process.num_threads()
            
            # Get file handles and network connections
            try:
                open_files = len(self.process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
            
            try:
                network_connections = len(self.process.connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                network_connections = 0
            
            return ProcessMetrics(
                timestamp=time.time(),
                pid=self.process.pid,
                cpu_percent=cpu_percent,
                memory_mb=memory_info.rss / 1024 / 1024,
                memory_percent=memory_percent,
                threads=threads,
                open_files=open_files,
                network_connections=network_connections
            )
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.error(f"Process metrics collection failed: {e}")
            # Return empty metrics
            return ProcessMetrics(
                timestamp=time.time(),
                pid=-1,
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
                threads=0,
                open_files=0,
                network_connections=0
            )
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check alert conditions and trigger alerts if needed"""
        
        current_time = time.time()
        
        for alert_rule in self.alert_rules:
            if not alert_rule.enabled:
                continue
            
            # Check cooldown
            if (alert_rule.last_triggered and 
                current_time - alert_rule.last_triggered < alert_rule.cooldown_seconds):
                continue
            
            try:
                # Check condition
                if alert_rule.condition(metrics):
                    # Trigger alert
                    self._trigger_alert(alert_rule, metrics)
                    alert_rule.last_triggered = current_time
                    
            except Exception as e:
                self.logger.error(f"Alert condition check failed for {alert_rule.name}: {e}")
    
    def _trigger_alert(self, alert_rule: AlertRule, metrics: SystemMetrics):
        """Trigger an alert"""
        
        # Format message with metrics
        message = alert_rule.message.format(**metrics.__dict__)
        
        alert_info = {
            "timestamp": time.time(),
            "name": alert_rule.name,
            "severity": alert_rule.severity,
            "message": message,
            "metrics": metrics.__dict__
        }
        
        # Add to alert history
        self.alerts_history.append(alert_info)
        
        # Keep only last 100 alerts
        if len(self.alerts_history) > 100:
            self.alerts_history = self.alerts_history[-100:]
        
        # Log alert
        log_level = getattr(logging, alert_rule.severity.upper(), logging.WARNING)
        self.logger.log(log_level, f"ALERT: {message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_info)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def add_alert_rule(self, alert_rule: AlertRule):
        """Add a custom alert rule"""
        self.alert_rules.append(alert_rule)
        self.logger.info(f"Added alert rule: {alert_rule.name}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system and process metrics"""
        
        system_metrics = self._collect_system_metrics()
        process_metrics = self._collect_process_metrics()
        
        return {
            "system": system_metrics.__dict__,
            "process": process_metrics.__dict__,
            "timestamp": time.time()
        }
    
    def get_metrics_summary(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """Get summary of metrics over specified duration"""
        
        cutoff_time = time.time() - (duration_minutes * 60)
        
        # Filter recent metrics
        recent_system = [m for m in self.system_metrics_history if m.timestamp >= cutoff_time]
        recent_process = [m for m in self.process_metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_system:
            return {"error": "No metrics available for specified duration"}
        
        # Calculate statistics
        system_stats = self._calculate_metrics_stats(recent_system)
        process_stats = self._calculate_metrics_stats(recent_process)
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts_history if a["timestamp"] >= cutoff_time]
        
        return {
            "duration_minutes": duration_minutes,
            "metrics_count": len(recent_system),
            "system": system_stats,
            "process": process_stats,
            "alerts": {
                "count": len(recent_alerts),
                "by_severity": self._group_alerts_by_severity(recent_alerts),
                "recent": recent_alerts[-5:] if recent_alerts else []
            }
        }
    
    def _calculate_metrics_stats(self, metrics: List) -> Dict[str, Any]:
        """Calculate statistics for a list of metrics"""
        
        if not metrics:
            return {}
        
        # Numeric fields to calculate stats for
        numeric_fields = [
            'cpu_percent', 'memory_percent', 'memory_available_mb', 
            'disk_usage_percent', 'gpu_usage_percent', 'gpu_memory_percent'
        ]
        
        stats = {}
        
        for field in numeric_fields:
            values = []
            for metric in metrics:
                value = getattr(metric, field, None)
                if value is not None:
                    values.append(value)
            
            if values:
                stats[field] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "current": values[-1] if values else None
                }
        
        return stats
    
    def _group_alerts_by_severity(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group alerts by severity level"""
        
        severity_counts = {}
        for alert in alerts:
            severity = alert["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return severity_counts
    
    def check_resource_requirements(self, operation: str, **requirements) -> Dict[str, Any]:
        """Check if system resources meet requirements for an operation"""
        
        current_metrics = self._collect_system_metrics()
        
        checks = {
            "memory_mb": requirements.get("memory_mb"),
            "cpu_percent_available": requirements.get("cpu_percent"),
            "disk_space_mb": requirements.get("disk_space_mb"),
            "gpu_memory_mb": requirements.get("gpu_memory_mb")
        }
        
        results = {
            "operation": operation,
            "can_proceed": True,
            "warnings": [],
            "blockers": [],
            "current_resources": current_metrics.__dict__
        }
        
        # Check memory
        if checks["memory_mb"]:
            available_mb = current_metrics.memory_available_mb
            required_mb = checks["memory_mb"]
            
            if available_mb < required_mb:
                results["can_proceed"] = False
                results["blockers"].append(
                    f"Insufficient memory: need {required_mb}MB, available {available_mb:.0f}MB"
                )
            elif available_mb < required_mb * 1.5:  # Less than 1.5x required
                results["warnings"].append(
                    f"Low memory margin: need {required_mb}MB, available {available_mb:.0f}MB"
                )
        
        # Check CPU
        if checks["cpu_percent_available"]:
            current_cpu = current_metrics.cpu_percent
            required_cpu = checks["cpu_percent_available"]
            available_cpu = 100 - current_cpu
            
            if available_cpu < required_cpu:
                results["warnings"].append(
                    f"High CPU usage: {current_cpu:.1f}% used, may impact performance"
                )
        
        # Check GPU memory
        if checks["gpu_memory_mb"] and current_metrics.gpu_memory_percent:
            # Estimate available GPU memory (this is simplified)
            gpu_memory_used_percent = current_metrics.gpu_memory_percent
            if gpu_memory_used_percent > 80:
                results["warnings"].append(
                    f"High GPU memory usage: {gpu_memory_used_percent:.1f}%"
                )
        
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment"""
        
        current_metrics = self._collect_system_metrics()
        recent_alerts = [a for a in self.alerts_history if time.time() - a["timestamp"] < 3600]
        
        # Health scoring (0-100)
        health_score = 100
        health_issues = []
        
        # CPU health
        if current_metrics.cpu_percent > 90:
            health_score -= 20
            health_issues.append("High CPU usage")
        elif current_metrics.cpu_percent > 70:
            health_score -= 10
            health_issues.append("Elevated CPU usage")
        
        # Memory health
        if current_metrics.memory_percent > 90:
            health_score -= 25
            health_issues.append("High memory usage")
        elif current_metrics.memory_percent > 80:
            health_score -= 15
            health_issues.append("Elevated memory usage")
        
        # Disk health
        if current_metrics.disk_usage_percent > 95:
            health_score -= 20
            health_issues.append("Critical disk space")
        elif current_metrics.disk_usage_percent > 85:
            health_score -= 10
            health_issues.append("Low disk space")
        
        # Alert frequency
        critical_alerts = [a for a in recent_alerts if a["severity"] == "critical"]
        if len(critical_alerts) > 5:
            health_score -= 30
            health_issues.append("Multiple critical alerts")
        elif len(recent_alerts) > 20:
            health_score -= 15
            health_issues.append("High alert frequency")
        
        # Temperature (if available)
        if current_metrics.temperature and current_metrics.temperature > 80:
            health_score -= 15
            health_issues.append("High temperature")
        
        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        elif health_score >= 40:
            status = "poor"
        else:
            status = "critical"
        
        return {
            "status": status,
            "score": max(0, health_score),
            "issues": health_issues,
            "metrics": current_metrics.__dict__,
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len(critical_alerts),
            "monitoring_active": self.is_monitoring
        }
    
    def export_metrics(self, filepath: str, duration_hours: int = 24):
        """Export metrics to file"""
        
        cutoff_time = time.time() - (duration_hours * 3600)
        
        export_data = {
            "export_timestamp": time.time(),
            "duration_hours": duration_hours,
            "system_metrics": [
                m.__dict__ for m in self.system_metrics_history 
                if m.timestamp >= cutoff_time
            ],
            "process_metrics": [
                m.__dict__ for m in self.process_metrics_history 
                if m.timestamp >= cutoff_time
            ],
            "alerts": [
                a for a in self.alerts_history 
                if a["timestamp"] >= cutoff_time
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")
    
    def __del__(self):
        """Cleanup when monitor is destroyed"""
        self.stop_monitoring()
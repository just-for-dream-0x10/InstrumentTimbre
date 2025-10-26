"""
System Integration Module

Provides unified integration layer for all InstrumentTimbre modules,
including error handling, module coordination, and system orchestration.
"""

from .integration_engine import IntegrationEngine
from .error_handler import SystemErrorHandler
from .module_coordinator import ModuleCoordinator
from .system_monitor import SystemMonitor
from .exception_types import *

__all__ = [
    'IntegrationEngine',
    'SystemErrorHandler', 
    'ModuleCoordinator',
    'SystemMonitor'
]
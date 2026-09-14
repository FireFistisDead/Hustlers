"""
PharmaGuard AI — Signal detection module.

Public API:
    from backend.app.signal import SignalEngine, SignalResult, SignalUnit, RoleScope

DISCLAIMER: All results are potential disproportionate reporting signals only.
Not confirmed safety signals. Not evidence of causality.
Requires clinical and regulatory expert review.
"""
from .engine import SignalEngine
from .models import MetricValue, RoleScope, SignalResult, SignalUnit, TwoByTwo

__all__ = [
    "SignalEngine",
    "SignalResult",
    "SignalUnit",
    "TwoByTwo",
    "MetricValue",
    "RoleScope",
]

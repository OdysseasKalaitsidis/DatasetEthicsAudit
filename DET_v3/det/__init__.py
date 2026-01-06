"""DET v3 Package - Dataset Ethics Triage for pre-training assessment."""

__version__ = '3.0.0'

from .metrics import (
    calculate_urs, calculate_aoi, calculate_dmi, calculate_k_anonymity, calculate_hrs,
    calculate_foi, calculate_fpc, calculate_cpa, calculate_spa, calculate_dai
)

from .decision import DETDecisionEngine, make_decision

__all__ = [
    'calculate_urs', 'calculate_aoi', 'calculate_dmi', 'calculate_k_anonymity', 'calculate_hrs',
    'calculate_foi', 'calculate_fpc', 'calculate_cpa', 'calculate_spa', 'calculate_dai',
    'DETDecisionEngine', 'make_decision'
]

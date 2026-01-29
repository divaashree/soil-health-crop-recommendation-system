"""
Soil Health Analytics - Core Modules
This file makes the src directory a Python package
"""

__version__ = "1.0.0"
__author__ = "Soil Health Analytics Team"

# Import key modules for easy access
try:
    from .predict import soil_predictor, crop_predictor
    __all__ = ['soil_predictor', 'crop_predictor']
except ImportError:
    # Modules not available yet
    pass

print(f"✅ Soil Health Analytics Package v{__version__} loaded")
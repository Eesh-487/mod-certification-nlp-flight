# Utils package for Aircraft Modification Certification Tool
# Contains utility modules for text processing, regulation mapping, and similarity analysis

__version__ = "2.0"
__author__ = "Aircraft Certification Team"

# Import main utility classes for easier access
try:
    from .preprocessing import TextPreprocessor, FeatureExtractor
    from .regulation_mapper import RegulationMapper
    from .similarity_engine import SimilarityEngine
    
    __all__ = [
        'TextPreprocessor',
        'FeatureExtractor', 
        'RegulationMapper',
        'SimilarityEngine'
    ]
    
except ImportError:
    # Graceful fallback if dependencies not available
    __all__ = []

"""
Path configurations for CV Classification Pipeline
"""

from pathlib import Path


class Paths:
    """Centralized path management"""
    
    # Root directory
    ROOT_DIR = Path(__file__).parent.parent
    
    # Data directories
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Output directories
    OUTPUTS_DIR = ROOT_DIR / "outputs"
    MODELS_DIR = OUTPUTS_DIR / "models"
    OOF_DIR = OUTPUTS_DIR / "oof"
    LOGS_DIR = OUTPUTS_DIR / "logs"
    SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
    
    # Source directory
    SRC_DIR = ROOT_DIR / "src"
    
    # Config directory
    CONFIG_DIR = ROOT_DIR / "config"
    
    # EDA directory
    EDA_DIR = ROOT_DIR / "eda"
    
    # Notebooks directory
    NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
    
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.OUTPUTS_DIR,
            cls.MODELS_DIR,
            cls.OOF_DIR,
            cls.LOGS_DIR,
            cls.SUBMISSIONS_DIR,
            cls.EDA_DIR,
            cls.NOTEBOOKS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

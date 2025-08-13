import sys
from pathlib import Path

def setup_paths():
    root = Path(__file__).parent.parent
    sys.path.extend([
        str(root),
        str(root / "polygon_processors"),
        str(root / "community_detection"),
        str(root / "access_metrics")
    ])

setup_paths()
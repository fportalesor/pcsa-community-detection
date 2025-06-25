from pathlib import Path

def get_paths():
    base_dir = Path(__file__).parent
    input_dir = base_dir / "data/raw"
    output_dir = base_dir / "data/processed"
    aztool_dir = base_dir / "AZTool"
    output_dir.mkdir(exist_ok=True)
    return input_dir, output_dir, aztool_dir

if __name__ == '__main__':
    get_paths()
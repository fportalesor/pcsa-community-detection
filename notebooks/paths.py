from pathlib import Path

def get_paths():
    base_dir = Path().cwd().parent
    input_dir = base_dir / "data/raw"
    output_dir = base_dir / "data/processed"
    plots_dir = base_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    return input_dir, output_dir, plots_dir

if __name__ == '__main__':
    get_paths()
from pathlib import Path
import argparse
from polygon_processors import (
    UrbanRuralPolygonMerger,
    MultipartPolygonProcessor)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process Chilean Census polygons data.")
    parser.add_argument('-u', '--urban', type=str, required=True,
                      help="Path to the input urban census polygons data file")
    parser.add_argument('-r', '--rural', type=str, required=True,
                      help="Path to the input rural census polygons data file")
    parser.add_argument('-o', '--output', type=str, default="relabelled_polygons.shp",
                      help="Output datafile with processed census polygons")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    base_dir = Path(__file__).parent
    
    input_dir = base_dir / "data/raw"
    # Ensure output directory exists
    output_dir = base_dir / "data/processed"
    output_dir.mkdir(exist_ok=True)

    # Workflow
    merger = UrbanRuralPolygonMerger()
    merged_data = merger.process(
        urban_path=str(input_dir / args.urban),
        rural_path=str(input_dir / args.rural)
    )
    
    multipart_processor = MultipartPolygonProcessor(input_data=merged_data)
    
    processed_data = multipart_processor._relabel_multipart_blocks()

    processed_data.to_file(str(output_dir / args.output))
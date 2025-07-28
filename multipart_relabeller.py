import argparse
from paths import get_paths
from polygon_processors import (
    UrbanRuralPolygonMerger,
    MultipartPolygonRelabeller)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process Chilean Census polygons data.")
    parser.add_argument('-u', '--urban', type=str, required=True,
                      help="Path to the input urban census polygons data file")
    parser.add_argument('-r', '--rural', type=str, required=True,
                      help="Path to the input rural census polygons data file")
    parser.add_argument('-o', '--output', type=str, default="processed_polygons.parquet",
                      help="Output datafile with processed census polygons")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    input_dir, output_dir, _ = get_paths()

    # Workflow
    merger = UrbanRuralPolygonMerger()
    merged_data = merger.process(
        urban_path=str(input_dir / args.urban),
        rural_path=str(input_dir / args.rural)
    )
    
    multipart_processor = MultipartPolygonRelabeller(input_data=merged_data)
    
    processed_data = multipart_processor._relabel_multipart_blocks()

    processed_data.to_parquet(str(output_dir / args.output))
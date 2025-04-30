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
    parser.add_argument('-o', '--output', type=str, default="processed_data.shp",
                      help="Output datafile with processed census polygons")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    # Get the directory containing polygon_processors package
    base_dir = Path(__file__).parent
    polygon_processors_dir = base_dir / "polygon_processors"
    
    input_dir = polygon_processors_dir / "data"
    # Ensure output directory exists
    output_dir = polygon_processors_dir / "processed_data"
    output_dir.mkdir(exist_ok=True)

    # Workflow
    merger = UrbanRuralPolygonMerger()
    merged_data = merger.process(
        manzanas_path=str(input_dir / args.urban),
        entidades_path=str(input_dir / args.rural)
    )
    
    multipart_processor = MultipartPolygonProcessor(merged_data)
    processed_data = multipart_processor.process()

    processed_data.to_file(str(output_dir / args.output))


# python multipart_processing.py -u 'manzanas_apc_2023.shp' -r 'microdatos_entidad.zip' -o 'processed_data.shp'
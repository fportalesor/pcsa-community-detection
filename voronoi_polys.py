from pathlib import Path
import argparse
import geopandas as gpd
import time

from polygon_processors import (
    VoronoiProcessor
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process Chilean Census polygons data.")
    parser.add_argument('-i', '--input', type=str, required=True,
                      help="Input datafile with processed census polygons")
    parser.add_argument('-r', '--regions', type=str, required=True,
                      help="Path to the input region polygons data file")
    parser.add_argument('-b', '--barriers', type=str,
                      help="Path to the input barrier mask data file")
    parser.add_argument('-l', '--region_list', type=int, nargs='+',
                      default=[13111, 13110, 13112, 13202, 13201, 13131, 13203],
                      help="List of region codes to process")
    parser.add_argument("--overlay-hidden", action="store_true",
                      help="If True, will perform an overlay between visible and hidden polygons (default: disabled).")
    parser.add_argument('-o', '--output', type=str, default="voronoi.gpkg",
                      help="Output datafile with the resulting voronoi polygons")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    # Get the directory containing polygon_processors package
    base_dir = Path(__file__).parent
    
    input_dir = base_dir / "data/raw"
    # Ensure output directory exists
    output_dir = base_dir / "data/processed"
    output_dir.mkdir(exist_ok=True)

    output_gpkg = output_dir / args.output

    start_time = time.time()

    processed_data = gpd.read_file(output_dir / args.input)
    
    for region in args.region_list:
        voronoi_processor = VoronoiProcessor(
            processed_data, 
            region_id=region,
            root_folder=output_dir
        )

        voronoi_data = voronoi_processor.process(
            region_path=input_dir / args.regions,
            barrier_mask_path=input_dir / args.barriers,
            overlay_hidden=args.overlay_hidden
        )

        # Write to GPKG, using region code as layer name
        voronoi_data.to_file(output_gpkg, layer=str(region), driver="GPKG")

    voronoi_gdf = voronoi_processor.combine_layers_from_gpkg(output_gpkg, args.region_list)

    voronoi_gdf.to_file(output_gpkg, layer="combined", driver="GPKG", mode="w")

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60

    print(f"Execution time: {duration_minutes:.2f} minutes")
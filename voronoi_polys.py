from pathlib import Path
import argparse
import geopandas as gpd
import time

from polygon_processors import (
    VoronoiProcessor
)

def parse_arguments():
    parser = argparse.ArgumentParser(description=(
        "Standalone workflow to generate and process Voronoi diagrams " 
        "from input polygons, including buffering, densification, hidden "
        "polygon handling, and boundary simplification within a target region."
        ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', type=str, required=True,
                      help="Input datafile with processed census polygons")
    parser.add_argument('-r', '--regions', type=str, required=True,
                      help="Path to the input region polygons data file")
    parser.add_argument('-b', '--barriers', type=str, default='hidrographic_network.shp',
                      help="Path to the input barrier mask data file")
    parser.add_argument('-l', '--region_list', type=int, nargs='+',
                      default=[13111, 13110, 13112, 13202, 13201, 13131, 13203],
                      help="List of region codes to process")
    parser.add_argument('-ir', '--intermediate-regions', type=str, default='ZONA_C17.shp',
                      help="File path to the polygon dataset representing " \
                      "intermediate administrative regions")
    parser.add_argument('--no-return-hidden', dest='return_hidden', action='store_false',
                    help="Disable returning both visible and hidden polygons.")
    parser.add_argument("--overlay-hidden", action="store_true",
                      help="If True, will perform an overlay between visible and hidden polygons.")
    parser.add_argument('-o', '--output', type=str, default='voronoi.gpkg',
                      help="Output datafile with the resulting voronoi polygons")
    parser.add_argument('-oh', '--output-hidden', type=str, default='hidden_polys.gpkg',
                      help="Output datafile with the resulting hidden voronoi polygons")
    parser.add_argument('--no-by-chunks', dest='by_chunks', action='store_false',
                    help="Disable processing regions by chunks")
    parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help="Disable verbose output.")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    # Get the directory containing polygon_processors package
    base_dir = Path(__file__).parent
    
    input_dir = base_dir / "data/raw"
    # Ensure output directory exists
    output_dir = base_dir / "data/processed"
    output_dir.mkdir(exist_ok=True)

    output_voronoi = output_dir / args.output
    output_hidden = output_dir / args.output_hidden

    start_time = time.time()

    processed_data = gpd.read_file(output_dir / args.input)
    
    for region in args.region_list:
        voronoi_processor = VoronoiProcessor(
            processed_data, 
            region_id=region
        )

        result = voronoi_processor.process(
            region_path=input_dir / args.regions,
            barrier_mask_path=input_dir / args.barriers,
            int_region_path= input_dir / args.intermediate_regions,
            overlay_hidden=args.overlay_hidden,
            verbose=args.verbose,
            return_hidden=args.return_hidden,
            by_chunks=args.by_chunks
        )

        if args.return_hidden:
            visible_polys, hidden_polys = result
            visible_polys.to_file(output_voronoi, layer=str(region), driver="GPKG")

            if not hidden_polys.empty:
                hidden_polys.to_file(output_hidden, layer=str(region), driver="GPKG")
        else:
            visible_polys = result
            visible_polys.to_file(output_voronoi, layer=str(region), driver="GPKG")

    if args.return_hidden:
        combined_visible = voronoi_processor.combine_layers_from_gpkg(output_voronoi, args.region_list)
        combined_visible.to_file(output_voronoi, layer="combined", driver="GPKG", mode="w")

        combined_hidden = voronoi_processor.combine_layers_from_gpkg(output_hidden, args.region_list)
        combined_hidden.to_file(output_hidden, layer="combined", driver="GPKG", mode="w")

    else:
        combined_visible = voronoi_processor.combine_layers_from_gpkg(output_voronoi, args.region_list)
        combined_visible.to_file(output_voronoi, layer="combined", driver="GPKG", mode="w")
        
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60

    print(f"Execution time: {duration_minutes:.2f} minutes")
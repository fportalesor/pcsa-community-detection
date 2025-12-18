import argparse
from paths import get_paths
from polygon_processors import (
    UrbanRuralPolygonMerger,
    MultipartPolygonRelabeller)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process Chilean Census polygons data.")
    parser.add_argument('-u', '--urban', type=str, default="Cartografía_censo2024_R13_Manzanas.parquet",
                      help="Path to the input urban census polygons data file")
    parser.add_argument('-r', '--rural', type=str, default="Cartografía_censo2024_R13_Entidades.parquet",
                      help="Path to the input rural census polygons data file")
    parser.add_argument('-pi', '--poly-id', type=str, default="block_id",
                      help="Path to the input rural census polygons data file")
    parser.add_argument('-lc', '--list-coms', type=int, nargs='+',
                        default=[13110, 13111, 13112, 13202, 13201, 13131, 13203],
                      help="List of commune IDs to include")
    parser.add_argument('-nc', '--num-cols', type=str, nargs='+',
                        default=["n_per", "n_vp_ocupada"],
                      help="List of numeric columns to include")
    parser.add_argument('-o', '--output', type=str, default="processed_polygons.parquet",
                      help="Output datafile with processed census polygons")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    input_dir, output_dir, _ = get_paths()

    # Workflow
    merger = UrbanRuralPolygonMerger(list_coms=args.list_coms,
                                     poly_id=args.poly_id,
                                     num_cols=args.num_cols)
    merged_data = merger.process(
        urban_path=str(input_dir / args.urban),
        rural_path=str(input_dir / args.rural)
    )
    
    multipart_processor = MultipartPolygonRelabeller(input_data=merged_data,
                                                     poly_id=args.poly_id)
    
    processed_data = multipart_processor._relabel_multipart_blocks()

    processed_data.to_parquet(str(output_dir / args.output))



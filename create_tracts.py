import time
import geopandas as gpd
import argparse
from paths import get_paths
from polygon_processors import PolygonDissolver

def parse_arguments():
    parser = argparse.ArgumentParser(description=(
        "Dissolve polygons into Tracts with the AZTool Outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i', '--input', type=str, required=True,
                      help="Input data file with building block polygons (e.g., 'voronoi_data.shp')")
    parser.add_argument('-t', '--target-pop', type=str, nargs='+',
                      default=['100', '150', '200', '250', '300', '350', '400', '450', '500',
                               '550', '600', '650', '700', '750'],
                      help="List of target population sizes for tract dissolution. Each population value " \
                        "corresponds to a CSV file named 'TractOutput_<pop>.csv' in the AZTool folder.")
    parser.add_argument('-azt', '--aztool-ids', type=str, required=True,
                      help="Filename of AZTool Building Block IDs (e.g., 'voronoi.pat')")
    parser.add_argument('-o', '--output', type=str, default="tracts.gpkg",
                      help="Output GeoPackage file to store dissolved tracts.")
    parser.add_argument('-so', '--summary-output', type=str, default="tracts_summary.xlsx",
                      help="Excel file to save summary statistics of dissolved tracts.")
    return parser.parse_args()

def main():

    args = parse_arguments()
    _ , output_dir, aztool_dir = get_paths()

    input_path = output_dir / args.input
    aztool_ids_path = aztool_dir / args.aztool_ids

    output_gpkg = output_dir / args.output
    excel_summary_path = output_dir / args.summary_output

    start_time = time.time()

    # Load input shapefile
    voronoi_data = gpd.read_file(input_path)

    # Initialise the dissolver
    dissolve_processor = PolygonDissolver(voronoi_data)

    stats_list = []

    for pop in args.target_pop:
        tract_csv_path = aztool_dir / f"TractOutput_{pop}.csv"
        dissolved_polys, stats = dissolve_processor.process_aztool_outcomes(tracts_path=tract_csv_path,
                                                                            aztool_ids_path=aztool_ids_path,
                                                                            fill_holes=True,
                                                                            remove_dangles=True)
        stats_list.append(stats)

        dissolved_polys.to_file(output_gpkg, layer=f"tracts_{pop}", driver="GPKG")

    if stats_list:
        summary_df = dissolve_processor._concat_stats(stats_list, "pop")
        summary_df.to_excel(excel_summary_path, index=False)

    print(f"\nProcessing complete in {time.time() - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
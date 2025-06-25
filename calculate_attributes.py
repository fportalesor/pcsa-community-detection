import argparse
import time
import geopandas as gpd
from paths import get_paths
from polygon_processors import (
    AttributeCalculator,
    PointCreator
)

def parse_arguments():
    parser = argparse.ArgumentParser(description=(
        "Workflow to calculate attributes for polygons, intended to be used as "
        "building blocks in the AZTool software for automated zone design."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-vi', '--voronoi-input', type=str, default="combined",
                      help="Path to input file containing Voronoi polygons.")
    parser.add_argument('-pi', '--points-input', type=str,
                      default="data.csv", help="Input data with latitude and longitude columns.")
    parser.add_argument('-pid', '--points-id', type=str, default="id",
                      help="Name of the column used as unique identifier in the points data.")
    parser.add_argument('-plid', '--polygons-id', type=str, default="block_id",
                      help="Name of the column used as unique identifier in the polygons data.")
    parser.add_argument('-sedu', '--sedata-urban', type=str, default="ISMT_2017_Zonas_Censales.zip",
                      help="Path to urban socioeconomic data file (ZIP).")
    parser.add_argument('-sedr', '--sedata-rural', type=str, default="ISMT_2017_Localidades_Rurales.zip",
                      help="Path to rural socioeconomic data file (ZIP).")
    parser.add_argument("--split-polygons", action="store_true",
                      help="Enable splitting of polygons exceeding the population threshold")
    parser.add_argument('-pm', '--pop-max', type=int, default=200,
                      help="Population threshold above which polygons will be split if splitting is enabled.")
    parser.add_argument('-t', '--tolerance', type=float, default=0.3,
                      help="Allowed proportional deviation from target population per cluster (e.g., 0.2 = ±20%%)")
    parser.add_argument('-b', '--buffer_radius', type=int, default=100,
                      help="Buffer radius for point movement in metres")
    parser.add_argument('-ot', '--overlap_threshold', type=int, default=10,
                      help="Minimum number of overlapping points to trigger movement")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    input_dir, output_dir, _ = get_paths()

    start_time = time.time()

    voronoi_gpkg_path = output_dir / "voronoi.gpkg"
    voronoi = gpd.read_file(voronoi_gpkg_path, layer=args.voronoi_input)

    point_creator = PointCreator(
        buffer_radius=args.buffer_radius,
        overlap_threshold=args.overlap_threshold
    )

    # Create points - conditionally apply movement
    if args.split_polygons:
        points = point_creator.create_points_from_file(
            input_dir / args.points_input,
            gdf_polygons=voronoi,
            points_id= args.points_id,
            move_points=True,
        )
        points.to_file(output_dir / "moved_points.shp")
    else:
        points = point_creator.create_points_from_file(
            input_dir / args.points_input,
            move_points=False
        )
        points.to_file(output_dir / "points.shp")
    
    attribute_calculator = AttributeCalculator(input_data=voronoi,
                                               points_id=args.points_id,
                                               split_polygons=args.split_polygons,
                                               pop_max=args.pop_max,
                                               tolerance=args.tolerance)
    
    voronoi_se_data = attribute_calculator.process(geocoded_data=points,
                                                  urban_se_data_path= input_dir /
                                                  args.sedata_urban,
                                                  rural_se_data_path= input_dir / 
                                                  args.sedata_rural)
    
    if args.split_polygons:
        voronoi_se_data.to_file(output_dir / "voronoi_data_split.shp")
    else:
        voronoi_se_data.to_file(output_dir / "voronoi_data.shp")

    split_polys, duplicates = attribute_calculator.identify_multipart_polygons(
        voronoi_se_data, args.polygons_id
    )

    # Check and export multipart geometries
    if not duplicates.empty:
        gpkg_path = output_dir / "voronoi.gpkg"
        duplicates.to_file(gpkg_path, layer="voronoi_duplicates", driver="GPKG", mode="w")

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60

    print(f"Execution time: {duration_minutes:.2f} minutes")
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

class PointCreator:
    """
    Creates point geometries from a CSV file.
    Optionally moves overlapping points if `move_points=True`.
    """

    @staticmethod
    def create_points_from_csv(point_data_path, move_points=False, gdf_polygons=None, 
                              buffer_radius=50, overlap_threshold=10, debug=False):
        """
        Loads points from CSV and optionally moves overlapping points.
        
        Parameters:
        - point_data_path (str): Path to CSV with 'id', 'latitude', 'longitude'.
        - move_points (bool): If True, moves overlapping points (requires gdf_polygons).
        - gdf_polygons (GeoDataFrame): Polygons to constrain movement (required if move_points=True).
        - buffer_radius (float): Radius for movement buffer (default: 50m).
        - overlap_threshold (int): Minimum overlap to trigger movement (default: 10).
        - debug (bool): Print debug info (default: False).

        Returns:
        - GeoDataFrame with points (in EPSG:32719) and 'was_moved' flag.
        """
        # Load and clean
        point_data = pd.read_csv(point_data_path, usecols=["id", "latitude", "longitude"])
        point_data = point_data.dropna(subset=["latitude", "longitude"]).drop_duplicates(subset="id")

        # Convert to GeoDataFrame
        gdf_points = gpd.GeoDataFrame(
            point_data,
            geometry=gpd.points_from_xy(point_data.longitude, point_data.latitude),
            crs="EPSG:4326"
        ).to_crs("EPSG:32719")

        # Update coordinates from geometry
        gdf_points["latitude"] = gdf_points.geometry.y
        gdf_points["longitude"] = gdf_points.geometry.x
        gdf_points["was_moved"] = False  # Initialise flag

        # Move points if requested
        if move_points:
            if gdf_polygons is None:
                raise ValueError("gdf_polygons must be provided if move_points=True")
            gdf_points = PointCreator.move_overlapping_points(
                gdf_points, gdf_polygons, buffer_radius, overlap_threshold, debug
            )

        return gdf_points

    @staticmethod
    def move_point_within_buffer(point, polygon, buffer_radius=50):
        """Move point randomly within buffer intersection with polygon."""
        buffer = point.buffer(buffer_radius).intersection(polygon)
        if buffer.is_empty:
            return point
        minx, miny, maxx, maxy = buffer.bounds
        for _ in range(100):  # Try 100 random positions
            candidate = Point(np.random.uniform(minx, maxx), 
                             np.random.uniform(miny, maxy))
            if buffer.contains(candidate):
                return candidate
        return point  # Fallback if no valid position found

    @staticmethod
    def move_overlapping_points(gdf_points, gdf_polygons, buffer_radius=50, 
                              overlap_threshold=10, debug=False):
        """
        Moves overlapping points within their polygons.
        """
        gdf_points = gdf_points.copy()
        original_geometries = gdf_points.set_index("id").geometry.copy()

        # Prepare polygons
        gdf_polygons = gdf_polygons.to_crs(gdf_points.crs)[["geometry"]]
        gdf_polygons["polygon_id"] = gdf_polygons.index

        # Spatial join (points within polygons)
        joined = gpd.sjoin(
            gdf_points[["id", "geometry"]],
            gdf_polygons,
            how="left",
            predicate="within"
        ).set_index("id")

        # Group points by rounded coordinates to detect overlaps
        gdf_points["x_rounded"] = gdf_points.geometry.x.round(4)
        gdf_points["y_rounded"] = gdf_points.geometry.y.round(4)
        gdf_points["group_id"] = (
            gdf_points["x_rounded"].astype(str) + "_" + gdf_points["y_rounded"].astype(str)
        )

        updated_geometries = {}

        for group_id, group in tqdm(gdf_points.groupby("group_id"), desc="Relocating clustered points"):
            if len(group) >= overlap_threshold:
                for _, row in group.iterrows():
                    pid = row["id"]
                    if pid not in joined.index:
                        if debug:
                            print(f"Point {pid} outside polygons")
                        continue
                    polygon_id = joined.loc[pid, "polygon_id"]
                    if pd.isna(polygon_id):
                        continue
                    polygon = gdf_polygons.loc[polygon_id, "geometry"]
                    moved = PointCreator.move_point_within_buffer(row.geometry, polygon, buffer_radius)
                    updated_geometries[pid] = moved

        # Apply changes and flag moved points
        gdf_points["geometry"] = gdf_points.apply(
            lambda row: updated_geometries.get(row["id"], row.geometry),
            axis=1
        )
        gdf_points["was_moved"] = gdf_points.apply(
            lambda row: row.geometry != original_geometries[row["id"]],
            axis=1
        )

        # Clean up and update coordinates
        gdf_points = gdf_points.drop(columns=["group_id", "x_rounded", "y_rounded"])
        gdf_points["latitude"] = gdf_points.geometry.y
        gdf_points["longitude"] = gdf_points.geometry.x

        # Summary
        total = len(gdf_points)
        moved = gdf_points["was_moved"].sum()
        print(f"\nMovement Summary:")
        print(f"Total points: {total}")
        print(f"Points moved: {moved}")

        return gdf_points
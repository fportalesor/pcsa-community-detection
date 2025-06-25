import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
from .base_processor import PolygonProcessor

class PointCreator(PolygonProcessor):
    """
    Creates spatial points from files and optionally relocates overlapping points 
    to avoid issues in downstream spatial operations.
    """

    def __init__(self, buffer_radius=50, overlap_threshold=10, crs_origin=4326, 
                 crs_destine=32719, debug=False, seed=7):
        self.buffer_radius = buffer_radius
        self.overlap_threshold = overlap_threshold
        self.crs_origin = crs_origin
        self.crs_destine = crs_destine
        self.debug = debug
        self.seed = seed
        """
        Initialise the point processing class with configuration parameters.

        Args:
            buffer_radius (float, optional): Movement buffer radius in meters. Defaults to 50.
            overlap_threshold (int, optional): Minimum number of overlapping points to trigger movement. Defaults to 10.
            crs_origin (int, optional): EPSG code for the source CRS (usually geographic). Defaults to 4326.
            crs_destine (int, optional): EPSG code for the target projected CRS (used for spatial operations). Defaults to 32719.
            debug (bool, optional): If True, enables debug output. Defaults to False.
            seed (int, optional): Random seed used for reproducibility in point displacement. Defaults to 7.
        """

    def create_points_from_file(self, point_data_path, gdf_polygons=None, 
                                points_id="id", move_points=False):
        """
        Load points from file and optionally move overlapping points.

        Args:
            point_data_path (str): Path to CSV or spatial file (must include point ID and coordinates or geometry).
            gdf_polygons (GeoDataFrame): Polygons to constrain movement (required if move_points=True).
            points_id (str): Column name to use as unique point identifier.
            move_points (bool, optional): Whether to move overlapping points within polygons. Defaults to False.

        Returns:
            GeoDataFrame with projected points and 'was_moved' flag.
        """
        if str(point_data_path).endswith(".csv"):
            # Check if required columns exist in CSV
            required_cols = [points_id, "latitude", "longitude"]
            sample = pd.read_csv(point_data_path, nrows=1)
            for col in required_cols:
                if col not in sample.columns:
                    raise ValueError(f"Missing required column '{col}' in CSV file.")
            
            point_data = pd.read_csv(point_data_path, usecols=[points_id, "latitude", "longitude"])
            point_data = point_data.dropna(subset=["latitude", "longitude"]).drop_duplicates(subset=points_id)
            gdf_points = gpd.GeoDataFrame(
                point_data,
                geometry=gpd.points_from_xy(point_data.longitude, point_data.latitude),
                crs=f"EPSG:{self.crs_origin}"
            )
            gdf_points = self._validate_crs(gdf_points, target_crs=self.crs_destine)
        else:
            gdf_points = gpd.read_file(point_data_path)
            if points_id not in gdf_points.columns:
                raise ValueError(f"'{points_id}' column not found in spatial file.")
            if "latitude" not in gdf_points.columns or "longitude" not in gdf_points.columns:
                raise ValueError("Spatial file must contain 'latitude' and 'longitude' columns.")
            if gdf_points.geometry.is_empty.any():
                raise ValueError("Some geometries are empty.")
            if gdf_points.crs is None:
                raise ValueError("Spatial file must have a defined CRS.")
            gdf_points = self._validate_crs(gdf_points, target_crs=self.crs_destine)

        # Update coordinates from geometry
        gdf_points["latitude"] = gdf_points.geometry.y
        gdf_points["longitude"] = gdf_points.geometry.x
        gdf_points["was_moved"] = False

        if move_points:
            if gdf_polygons is None:
                raise ValueError("gdf_polygons must be provided if move_points=True")
            gdf_points = self.move_overlapping_points(gdf_points, gdf_polygons, points_id)

        return gdf_points

    def move_point_within_buffer(self, point, polygon, buffer_radius=50, rng=None):
        """Move point randomly within buffer intersection with polygon."""
        if rng is None:
            rng = np.random.default_rng()
        buffer = point.buffer(buffer_radius).intersection(polygon)
        if buffer.is_empty:
            return point
        minx, miny, maxx, maxy = buffer.bounds
        for _ in range(100):
            candidate = Point(rng.uniform(minx, maxx),
                               rng.uniform(miny, maxy))
            if buffer.contains(candidate):
                return candidate
        return point

    def move_overlapping_points(self, gdf_points, gdf_polygons, points_id="id"):
        """
        Moves overlapping points within their polygons.
        """
        gdf_points = gdf_points.copy()
        original_geometries = gdf_points.set_index(points_id).geometry.copy()

        # Prepare polygons
        gdf_polygons = gdf_polygons.to_crs(gdf_points.crs)[["geometry"]]
        gdf_polygons["polygon_id"] = gdf_polygons.index

        # Spatial join (points within polygons)
        joined = gpd.sjoin(
            gdf_points[[points_id, "geometry"]],
            gdf_polygons,
            how="left",
            predicate="within"
        ).set_index(points_id)

        # Group points by rounded coordinates to detect overlaps
        gdf_points["x_rounded"] = gdf_points.geometry.x.round(4)
        gdf_points["y_rounded"] = gdf_points.geometry.y.round(4)
        gdf_points["group_id"] = (
            gdf_points["x_rounded"].astype(str) + "_" + gdf_points["y_rounded"].astype(str)
        )

        updated_geometries = {}
        rng = np.random.default_rng(self.seed)

        for _, group in tqdm(gdf_points.groupby("group_id"), desc="Relocating clustered points"):
            if len(group) >= self.overlap_threshold:
                for _, row in group.iterrows():
                    pid = row[points_id]
                    if pid not in joined.index:
                        if self.debug:
                            print(f"Point {pid} outside polygons")
                        continue
                    
                    polygon_ids = joined.loc[pid, "polygon_id"]

                    if not isinstance(polygon_ids, pd.Series):
                        polygon_ids = pd.Series([polygon_ids])

                    if polygon_ids.isna().all():
                        continue

                    polygon_id = polygon_ids.dropna().iloc[0]
                    polygon = gdf_polygons.loc[polygon_id, "geometry"]

                    moved = self.move_point_within_buffer(row.geometry, polygon, self.buffer_radius, rng)
                    updated_geometries[pid] = moved

        # Apply changes and flag moved points
        gdf_points["geometry"] = gdf_points.apply(
            lambda row: updated_geometries.get(row["id"], row.geometry),
            axis=1
        )
        gdf_points["was_moved"] = gdf_points.apply(
            lambda row: row.geometry != original_geometries[row[points_id]],
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
        print(f"Total points: {total:,}")
        print(f"Points moved: {moved:,}")

        return gdf_points
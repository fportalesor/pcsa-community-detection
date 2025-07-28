import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
from joblib import Parallel, delayed
from .base_processor import PolygonProcessor

class PointCreator(PolygonProcessor):
    """
    Creates spatial points from files and optionally relocates overlapping points 
    to avoid issues in downstream spatial operations.
    """

    def __init__(self, buffer_radius=50, overlap_threshold=10, crs_origin=4326, 
                 crs_destine=32719, debug=False, seed=7, n_jobs=1, chunks=20):
        self.buffer_radius = buffer_radius
        self.overlap_threshold = overlap_threshold
        self.crs_origin = crs_origin
        self.crs_destine = crs_destine
        self.debug = debug
        self.seed = seed
        self.n_jobs = n_jobs
        self.chunks = chunks
        """
        Initialise the point processing class with configuration parameters.

        Args:
            buffer_radius (float, optional): Movement buffer radius in meters. Defaults to 50.
            overlap_threshold (int, optional): Minimum number of overlapping points to trigger movement. Defaults to 10.
            crs_origin (int, optional): EPSG code for the source CRS (usually geographic). Defaults to 4326.
            crs_destine (int, optional): EPSG code for the target projected CRS (used for spatial operations). Defaults to 32719.
            debug (bool, optional): If True, enables debug output. Defaults to False.
            seed (int, optional): Random seed used for reproducibility in point displacement. Defaults to 7.
            n_jobs (int, optional): Number of CPU cores to use for parallel processing. Defaults to 1 (no parallel processing).
            chunks (int, optional): Number of chunks to divide polygons into for parallel processing. Defaults to 20.
        """

    def create_points_from_file(self, point_data_path, gdf_polygons=None, 
                                points_id="id", move_points=False):
        """
        Load points from CSV, Parquet, or spatial file and optionally move overlapping points.

        Args:
            point_data_path (str): Path to CSV, Parquet, or spatial file (must include point ID and coordinates or geometry).
            gdf_polygons (GeoDataFrame): Polygons to constrain movement (required if move_points=True).
            points_id (str): Column name to use as unique point identifier.
            move_points (bool, optional): Whether to move overlapping points within polygons. Defaults to False.

        Returns:
            GeoDataFrame with projected points and 'was_moved' flag.
        """
        if str(point_data_path).endswith(".csv"):
            required_cols = [points_id, "latitude", "longitude"]
            sample = pd.read_csv(point_data_path, nrows=1)
            for col in required_cols:
                if col not in sample.columns:
                    raise ValueError(f"Missing required column '{col}' in CSV file.")

            point_data = pd.read_csv(point_data_path, usecols=required_cols)
        
        elif str(point_data_path).endswith(".parquet"):
            point_data = pd.read_parquet(point_data_path)
            required_cols = [points_id, "latitude", "longitude"]
            for col in required_cols:
                if col not in point_data.columns:
                    raise ValueError(f"Missing required column '{col}' in Parquet file.")
            point_data = point_data[required_cols]
        
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

        if 'gdf_points' not in locals():  # for CSV or Parquet
            gdf_points = gpd.GeoDataFrame(
                point_data,
                geometry=gpd.points_from_xy(point_data.longitude, point_data.latitude),
                crs=f"EPSG:{self.crs_origin}"
            )
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

    def _process_polygon_chunk(self, chunk_polygons, points_in_polygons, original_points, points_id):
        """Process a chunk of polygons and their associated points."""
        chunk_geometries = {}
        
        for polygon_id, polygon in chunk_polygons.iterrows():
            # Get points for this polygon
            polygon_points = points_in_polygons.loc[
                points_in_polygons['polygon_id'] == polygon_id
            ].copy()
            
            if len(polygon_points) == 0:
                continue
                
            rng = np.random.default_rng(self.seed + polygon_id)
            
            polygon_points = polygon_points.assign(
                x_rounded=lambda x: x.geometry.x.round(4),
                y_rounded=lambda x: x.geometry.y.round(4)
            )
            polygon_points['group_id'] = (
                polygon_points['x_rounded'].astype(str) + "_" + 
                polygon_points['y_rounded'].astype(str)
            )

            for group_id, group in polygon_points.groupby("group_id"):
                if len(group) >= self.overlap_threshold:
                    for _, row in group.iterrows():
                        pid = row[points_id]
                        moved = self.move_point_within_buffer(
                            row.geometry, 
                            polygon.geometry, 
                            self.buffer_radius, 
                            rng
                        )
                        chunk_geometries[pid] = moved
        
        return chunk_geometries

    def move_overlapping_points(self, gdf_points, gdf_polygons, points_id="id"):
        """
        Moves overlapping points within their polygons using parallel processing by polygon chunks.
        """
        gdf_points = gdf_points.copy()
        original_geometries = gdf_points.set_index(points_id).geometry.copy()

        # Prepare polygons and ensure correct CRS
        gdf_polygons = gdf_polygons.to_crs(gdf_points.crs)[["geometry"]].copy()
        gdf_polygons["polygon_id"] = gdf_polygons.index

        # Spatial join to find which points are in which polygons
        joined = gpd.sjoin(
            gdf_points[[points_id, "geometry"]].copy(),
            gdf_polygons.copy(),
            how="left",
            predicate="within"
        )
        
        joined = joined[[points_id, 'geometry', 'polygon_id']].copy()

        # Split polygons into chunks for parallel processing
        n_chunks = min(self.chunks, len(gdf_polygons))
        indices = [
            (i * len(gdf_polygons)) // n_chunks for i in range(n_chunks + 1)
        ]

        polygon_chunks = [
            gdf_polygons.iloc[indices[i]:indices[i+1]]
            for i in range(n_chunks)
        ]
        
        # Prepare arguments for parallel processing
        args_list = []
        for chunk in polygon_chunks:
            # Get points that fall within any polygon in this chunk
            chunk_polygon_ids = chunk['polygon_id'].unique().tolist()
            chunk_points = joined.loc[joined['polygon_id'].isin(chunk_polygon_ids)].copy()
            args_list.append((
                chunk.copy(), 
                chunk_points.copy(), 
                original_geometries.copy(), 
                points_id
            ))

        # Process chunks in parallel or sequentially
        if self.n_jobs == 1:
            updated_geometries = {}
            for args in tqdm(args_list, desc="Relocating clustered points"):
                result = self._process_polygon_chunk(*args)
                updated_geometries.update(result)
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_polygon_chunk)(*args)
                for args in tqdm(args_list, desc="Relocating clustered points (parallel)")
            )
            updated_geometries = {}
            for result in results:
                updated_geometries.update(result)

        moved_mask = gdf_points[points_id].isin(updated_geometries.keys())
        gdf_points.loc[moved_mask, 'geometry'] = gdf_points.loc[moved_mask, points_id].map(updated_geometries)
        gdf_points['was_moved'] = moved_mask

        # Update coordinates
        gdf_points.loc[:, 'latitude'] = gdf_points.geometry.y
        gdf_points.loc[:, 'longitude'] = gdf_points.geometry.x

        # Summary
        total = len(gdf_points)
        moved = gdf_points["was_moved"].sum()
        print(f"\nMovement Summary:")
        print(f"Total points: {total:,}")
        print(f"Points moved: {moved:,}")

        return gdf_points
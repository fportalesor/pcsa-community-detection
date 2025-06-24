import numpy as np
import pandas as pd
import h3pandas
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from k_means_constrained import KMeansConstrained
from .base_processor import PolygonProcessor
from .hidden_polys import HiddenPolygonProcessor


class PolygonSplitter(PolygonProcessor):
    """
    Splits polygons with high population counts into smaller, spatially-constrained sub-polygons.
    """

    def __init__(self, input_data=None, points_id="id", poly_id="block_id", 
                 pop_max=150, tolerance=0.2, cluster_col="cluster",
                 resolution=12, buffer_distance=400, target_crs=32719, n_jobs=8, verbose=True):
        """
        Args:
        input_data (GeoDataFrame): Polygon input data with 'pop' attribute.
        points_id (str): Column name identifying input points.
        poly_id (str): Column name identifying polygons.
        pop_max (int): Population threshold for splitting.
        tolerance (float): Allowed variation in population size across clusters.
        cluster_col (str): Column name to assign cluster IDs.
        resolution (int): H3 resolution used for grid generation.
        buffer_distance (float): Distance to buffer polygons when generating H3 grids.
        target_crs (int): Target EPSG code for spatial operations.
        n_jobs (int): Number of parallel jobs.
        verbose (bool): Flag to enable progress messages.
        """
        self.data = input_data
        self.pop_max = pop_max
        self.points_id = points_id
        self.poly_id = poly_id
        self.cluster_col = cluster_col
        self.tolerance = tolerance
        self.resolution = resolution
        self.buffer_distance = buffer_distance
        self.target_crs = target_crs
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.hidden_processor = HiddenPolygonProcessor()

    def _split_building_blocks(self, points):
        """
        Splits all polygons in `self.data` with population above `pop_max`.

        Args:
            points (GeoDataFrame): Point data used for clustering within polygons.

        Returns:
            GeoDataFrame: The updated polygon layer with split and unsplit polygons combined.
        """        
        to_split = self.data[self.data["pop"] > self.pop_max].copy()
        if len(to_split) == 0:
            return self.data

        if self.verbose:
            print(f"\nNumber of Building Blocks to split: {len(to_split)}")

        building_block_ids = to_split[self.poly_id].unique()

        points_by_block = {
            block_id: points[points.geometry.within(self.data.loc[self.data[self.poly_id] == block_id].geometry.iloc[0])]
            for block_id in building_block_ids
        }

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._split_single_building_block)(points_by_block[block_id], block_id)
            for block_id in building_block_ids
        )

        results = [r for r in results if r is not None]
        split_polys = pd.concat(results, ignore_index=True)

        split_polys, duplicates = self.identify_multipart_polygons(split_polys, self.poly_id, keep_largest=False)
        if not duplicates.empty and self.verbose:
            print(f"Warning: {len(duplicates)} multipart polygons still remain after processing.")
        else:
            print("No multipart polygons found.")

        remaining_polys = self.data[~self.data[self.poly_id].isin(to_split[self.poly_id])].copy()
        remaining_polys = remaining_polys.drop(columns="pop")

        self.data = pd.concat([remaining_polys, split_polys], ignore_index=True).copy()
        self.data[self.poly_id] = self.data[self.poly_id].str.ljust(18, fillchar='0')

        return self.data

    def _split_single_building_block(self, points, building_block_id):
        """
        Splits a single polygon into sub-polygons using constrained clustering and hex aggregation.

        Args:
            points (GeoDataFrame): Points located within the polygon.
            building_block_id (str): ID of the polygon being split.

        Returns:
            GeoDataFrame: Subdivided polygon geometries with inherited attributes.
        """
        if self.verbose:
            print(f"Processing Building Block: {building_block_id}")

        poly = self.data[self.data[self.poly_id] == building_block_id].copy()
        n_clusters, pop_min, pop_max = self._calculate_pop_thresholds(poly)
        points_clustered = self._cluster_points(points, n_clusters, pop_min, pop_max)
        clusters_centroids = self._generate_cluster_centroids(points_clustered)

        h3_grid, counts = self._generate_h3_grid_and_counts(points_clustered, poly)
        dissolved = self._assign_clusters_to_hexes(h3_grid, counts, cluster_centroids_gdf=clusters_centroids)

        exploded = dissolved.explode(index_parts=False).reset_index(drop=True).copy()
        if exploded.duplicated(self.cluster_col, keep=False).any() and self.verbose:
            print(f"Fixing disconnected clusters...")
            dissolved = self.reassign_disconnected_parts(dissolved)

        dissolved[self.poly_id] = building_block_id
        for col in ["commune_id", "commune", "zone_type", "sregion_id"]:
            dissolved[col] = poly[col].iloc[0]

        dissolved[self.cluster_col] = dissolved[self.cluster_col].astype(str).str.zfill(2)
        dissolved[self.poly_id] = dissolved[self.poly_id] + dissolved[self.cluster_col]

        return dissolved[['commune_id', 'commune', self.poly_id, 'zone_type', 'sregion_id', 'geometry']]
    
    def _calculate_pop_thresholds(self, poly):
        """
        Calculates the number of clusters and the min/max population constraints per cluster.

        Args:
            poly (GeoDataFrame): A single polygon row with a 'pop' column.

        Returns:
            Tuple[int, int, int]: (number of clusters, min pop per cluster, max pop per cluster)
        """
        population = poly["pop"].iloc[0]
        n_clusters = max(round(population / self.pop_max), 2)
        pop_min = round((population / n_clusters) * (1 - self.tolerance))
        pop_max = round((population / n_clusters) * (1 + self.tolerance))
        return n_clusters, pop_min, pop_max

    def _cluster_points(self, points, n_clusters, pop_min, pop_max):
        """
        Applies size-constrained k-means clustering to points.

        Args:
            points (GeoDataFrame): Points to cluster.
            n_clusters (int): Desired number of clusters.
            pop_min (int): Minimum population per cluster.
            pop_max (int): Maximum population per cluster.

        Returns:
            GeoDataFrame: Points with cluster labels assigned.
        """
        if len(points) == 0:
            return None

        X = np.column_stack((points.geometry.x, points.geometry.y))

        if len(points) >= n_clusters:
            kmeans = KMeansConstrained(
                n_clusters=n_clusters,
                size_min=pop_min,
                size_max=pop_max,
                random_state=42
            )
            points[self.cluster_col] = kmeans.fit_predict(X) + 1
        else:
            points[self.cluster_col] = 0

        return points.drop(columns=[col for col in ["index_left", "index_right"] if col in points.columns])

    def _generate_cluster_centroids(self, points_gdf):
        """
        Computes centroids of each point cluster.

        Args:
            points_gdf (GeoDataFrame): Points with assigned clusters.

        Returns:
            GeoDataFrame: One point per cluster representing its centroid.
        """
        grouped = points_gdf.dissolve(by=self.cluster_col).reset_index()
        grouped['geometry'] = grouped.centroid
        grouped[self.cluster_col] = grouped[self.cluster_col].astype(str)
        return grouped[[self.cluster_col, 'geometry']]

    def _create_h3_grid(self, clip_polygon):
        """
        Generates an H3 hex grid over a buffered version of the given polygon.
        The polygon is buffered by `buffer_distance` to ensure the hexagonal tessellation
        fully covers the original polygon area, preventing gaps near polygon edges.

        Args:
            clip_polygon (GeoDataFrame): Polygon to use as base for the H3 grid.

        Returns:
            GeoDataFrame: H3 grid clipped to the original polygon extent.
        """
        poly_buffered = clip_polygon.copy()
        poly_buffered.geometry = poly_buffered.geometry.buffer(self.buffer_distance)
        # H3 requires geometries in EPSG:4326
        poly_buffered = poly_buffered.to_crs(4326)

        h3_grid = poly_buffered.h3.polyfill(resolution=self.resolution, explode=True)
        h3_grid = h3_grid.rename(columns={"h3_polyfill": "h3_index"}).set_index("h3_index")
        h3_grid = h3_grid.h3.h3_to_geo_boundary()
        h3_grid = self._validate_crs(h3_grid)
        h3_grid = h3_grid.clip(clip_polygon)

        return h3_grid

    def _count_points_per_hex(self, points_gdf):
        """
        Assigns H3 index to points and counts points per hexagon and cluster.
        """
        if self.cluster_col not in points_gdf.columns:
            raise ValueError(f"'{self.cluster_col}' column not found in points_gdf")

        points_wgs = points_gdf.to_crs(epsg=4326).copy()
        points_wgs = points_wgs.h3.geo_to_h3(resolution=self.resolution)
        h3_col = f"h3_{self.resolution}"

        grouped = points_wgs.groupby([h3_col, self.cluster_col]).size().reset_index(name='count')
        return grouped

    def _generate_h3_grid_and_counts(self, points_gdf, clip_polygon):
        """
        Wrapper to create hex grid and count points.
        """
        h3_grid = self._create_h3_grid(clip_polygon)
        grouped = self._count_points_per_hex(points_gdf)
        return h3_grid, grouped

    def _prepare_cluster_assignment(self, grouped):
        """
        Converts grouped counts into pivot table for cluster decision logic.
        """
        pivot = grouped.pivot(index=grouped.columns[0], columns=self.cluster_col, values='count').fillna(0)
        cluster_columns = grouped[self.cluster_col].unique()

        pivot['max_count'] = pivot[cluster_columns].max(axis=1)
        pivot['is_tie'] = pivot[cluster_columns].eq(pivot['max_count'], axis=0).sum(axis=1) > 1
        pivot[self.cluster_col] = pivot[cluster_columns].idxmax(axis=1)
        pivot.loc[pivot['is_tie'], self.cluster_col] = np.nan

        return pivot[[self.cluster_col]]

    def _resolve_cluster_assignment(self, h3_grid, pivot_assignments, cluster_centroids_gdf=None):
        """
        Resolves cluster assignments for hexes with ties or no dominant cluster,
        using nearest cluster centroid.

        Args:
            h3_grid (GeoDataFrame): The H3 cells.
            pivot_assignments (DataFrame): Cluster assignments with potential NaNs.
            cluster_centroids_gdf (GeoDataFrame): Fallback centroids for assignment.

        Returns:
            GeoDataFrame: H3 cells with final cluster assignments.
        """
        hex_gdf = h3_grid.merge(pivot_assignments, left_on='h3_index', right_index=True, how='left')

        if hex_gdf[self.cluster_col].isna().sum() > 0:
            if cluster_centroids_gdf is None:
                raise ValueError("Provide cluster_centroids_gdf for unassigned hexes.")

            centroids_utm = cluster_centroids_gdf.to_crs(epsg=self.target_crs)
            hex_gdf = hex_gdf.to_crs(epsg=self.target_crs)

            cent_coords = np.array(list(zip(centroids_utm.geometry.x, centroids_utm.geometry.y)))
            tree = cKDTree(cent_coords)

            empty_hexes = hex_gdf[hex_gdf[self.cluster_col].isna()].copy()
            hex_centroids = empty_hexes.geometry.centroid
            hex_coords = np.array(list(zip(hex_centroids.x, hex_centroids.y)))

            _, idx = tree.query(hex_coords)
            hex_gdf.loc[hex_gdf[self.cluster_col].isna(), self.cluster_col] = centroids_utm.iloc[idx][self.cluster_col].values

        return hex_gdf
    
    def _assign_clusters_to_hexes(self, h3_grid, grouped, cluster_centroids_gdf=None):
        """
        Assigns clusters to hex grid using prepared counts and strategy, and dissolves by cluster.
        """
        pivot_assignments = self._prepare_cluster_assignment(grouped)
        assigned = self._resolve_cluster_assignment(h3_grid, pivot_assignments, cluster_centroids_gdf)
        assigned[self.cluster_col] = assigned[self.cluster_col].astype(int).astype(str)
        dissolved = assigned.dissolve(by=self.cluster_col).reset_index().to_crs(epsg=self.target_crs)
        return dissolved
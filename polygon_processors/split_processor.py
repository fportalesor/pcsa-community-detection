import numpy as np
import pandas as pd
import geopandas as gpd
from k_means_constrained import KMeansConstrained
from shapely.geometry import Polygon, MultiPolygon
from longsgis import voronoiDiagram4plg
from .base_processor import PolygonProcessor
from .hidden_polys import HiddenPolygonProcessor

class PolygonSplitter(PolygonProcessor):
    """
    A class to split polygons with high population counts into smaller sub-polygons 
    using spatially-constrained clustering and Voronoi diagrams.
    
    Args:
        input_data (GeoDataFrame): Polygons to be split.
        id_points (str): Unique ID field for geocoded points.
        poly_id (str): Unique ID field for polygons.
        pop_max (int): Maximum population allowed per resulting cluster.
        split_polygons (bool): Whether to split polygons or not.
        tolerance (float): Allowable deviation from pop_max for clustering.
        root_folder (str or Path): Path to root working directory.
    """
    def __init__(self, input_data=None, id_points="id", poly_id="block_id", 
                 pop_max=150, split_polygons=False, 
                 tolerance=0.2, root_folder=None):
        super().__init__(root_folder)
        self.data = input_data
        self.pop_max = pop_max
        self.id_points = id_points
        self.poly_id = poly_id
        self.split_polygons = split_polygons
        self.tolerance = tolerance
        self.hidden_processor = HiddenPolygonProcessor()
    
    def _split_single_building_block(self, points, building_block_id):
        """
        Splits a single polygon into constrained clusters and generates Voronoi polygons.

        Args:
            points (GeoDataFrame): Geocoded points for clustering.
            building_block_id (str): Identifier of the polygon to split.

        Returns:
            GeoDataFrame or None: New clustered polygons or None if no points found.
        """

        poly = self.data[self.data[self.poly_id] == building_block_id].copy()
        points = points.sjoin(poly, how="inner", predicate="intersects")
        if len(points) == 0:
            return None
    
        X = np.column_stack((points.geometry.x, points.geometry.y))
        population = poly["pop"].iloc[0]
        n_clusters = max(round(population / self.pop_max), 2)

        pop_min = round((population/n_clusters) * (1 - self.tolerance))
        pop_max = round((population/n_clusters) * (1 + self.tolerance))
    
        if len(points) >= n_clusters:
            kmeans = KMeansConstrained(
                n_clusters=n_clusters,
                size_min=pop_min,       
                size_max=pop_max,  
                random_state=42
            )
            points["cluster"] = kmeans.fit_predict(X) + 1
        else:
            
            points["cluster"] = 0
    
        for col in ["index_left", "index_right"]:
            if col in points.columns:
                points = points.drop(columns=[col])

        vd = voronoiDiagram4plg(points, poly)
        vd = vd[["geometry", "cluster"]]
        dissolved = vd.dissolve(by="cluster").reset_index()

        exploded = dissolved.explode(index_parts=False).reset_index(drop=True).copy()
        duplicates = exploded[exploded.duplicated('cluster', keep=False)]
        
        if not duplicates.empty:
            print(f"Fixing disconnected clusters...")
            dissolved = self.fix_disconnected_clusters(dissolved)

        dissolved = self.assign_unclustered_parts(poly, dissolved, 1)

        dissolved[self.poly_id] = building_block_id
        for col in ["commune_id", "commune", "zone_type", "sregion_id"]:
            dissolved[col] = poly[col].iloc[0]

        dissolved["cluster"] = dissolved["cluster"].astype(str).str.zfill(2)
        dissolved[self.poly_id] = dissolved[self.poly_id] + dissolved["cluster"]

        return dissolved[['commune_id', 'commune', self.poly_id, 'zone_type', 'sregion_id', 'geometry']]
    
    def fix_disconnected_clusters(self, dissolved_gdf):
        """
        Reassigns disconnected parts of clusters to neighboring clusters with longest shared boundaries.

        Args:
            dissolved_gdf (GeoDataFrame): Clustered Voronoi polygons.

        Returns:
            GeoDataFrame: Fixed polygons with contiguous cluster geometries.
        """

        exploded = dissolved_gdf.explode(index_parts=False).reset_index(drop=True).copy()
        exploded["area"] = exploded.geometry.area

        duplicated_mask = exploded.duplicated("cluster", keep=False)
        duplicated_clusters = exploded[duplicated_mask].copy()
        idx_largest = duplicated_clusters.groupby("cluster")["area"].idxmax()
        largest_parts = exploded.loc[idx_largest]
        non_largest_parts = duplicated_clusters.drop(index=idx_largest).copy()
        non_largest_parts["original_index"] = non_largest_parts.index

        if len(non_largest_parts) == 0:
            return dissolved_gdf
    
        unique_clusters = non_largest_parts["cluster"].unique()

        # Simplified case: only two clusters exist
        if len(unique_clusters) == 2:
            cluster1, cluster2 = unique_clusters
        
            # For each small part, assign to the opposite cluster
            for idx, row in non_largest_parts.iterrows():
                current_cluster = row["cluster"]
                new_cluster = cluster2 if current_cluster == cluster1 else cluster1
                non_largest_parts.at[idx, "cluster"] = new_cluster

        else:
            # Logic for cases with more than 2 clusters
            potential_neighbors = exploded[
                ~exploded["cluster"].isin(non_largest_parts["cluster"])
            ].copy()

            for idx, row in non_largest_parts.iterrows():
                current_geom = row.geometry
                current_cluster = row["cluster"]
        
                touching = potential_neighbors[
                    potential_neighbors.geometry.touches(current_geom) | 
                    potential_neighbors.geometry.intersects(current_geom)
                ]
        
                if len(touching) == 0:
                    continue
            
                boundary_lengths = []
                for n_idx, neighbor in touching.iterrows():
                    intersection = current_geom.intersection(neighbor.geometry)
                    boundary_lengths.append((n_idx, intersection.length))
        
                boundary_lengths.sort(key=lambda x: x[1], reverse=True)
                if boundary_lengths:
                    best_neighbor_idx = boundary_lengths[0][0]
                    new_cluster = potential_neighbors.loc[best_neighbor_idx, "cluster"]
                    non_largest_parts.at[idx, "cluster"] = new_cluster

        cleaned = pd.concat([
            largest_parts.drop(columns=["area"], errors="ignore"),
            non_largest_parts.drop(columns=["area", "original_index", "centroid"], errors="ignore"),
            exploded[~exploded.index.isin(non_largest_parts.index) & 
                    ~exploded.index.isin(largest_parts.index)]
        ], ignore_index=True)

        return cleaned.dissolve(by="cluster").reset_index()

    def assign_unclustered_parts(self, original_poly, dissolved_gdf, area_threshold):
        """
        Assign leftover geometries (difference between original polygon and dissolved clusters)
        to the cluster that shares the longest boundary.
    
        Args:
            original_poly (GeoDataFrame): Original polygon geometry.
            dissolved_gdf (GeoDataFrame): Clustered Voronoi polygons.
            area_threshold (float): Minimum area to consider for assignment.
    
        Returns:
            GeoDataFrame: Updated cluster geometries with leftover areas assigned.
        """
        
        leftover_geom = original_poly.geometry.iloc[0].difference(dissolved_gdf.unary_union)
    
        if leftover_geom.is_empty or leftover_geom.area < area_threshold:
            return dissolved_gdf

        if isinstance(leftover_geom, Polygon):
            leftover_parts = [leftover_geom]
        elif isinstance(leftover_geom, MultiPolygon):
            leftover_parts = list(leftover_geom.geoms)
        else:
            return dissolved_gdf

        additions = []
        for part in leftover_parts:
            max_shared_length = 0
            best_cluster = None
        
            for _, row in dissolved_gdf.iterrows():
                intersection = part.intersection(row.geometry)
                if not intersection.is_empty:
                    shared_length = intersection.length
                    if shared_length > max_shared_length:
                        max_shared_length = shared_length
                        best_cluster = row["cluster"]

            if best_cluster is not None:
                additions.append({
                    "geometry": part,
                    "cluster": best_cluster
                })

        if not additions:
            return dissolved_gdf

        additions_gdf = gpd.GeoDataFrame(additions, geometry="geometry", crs=dissolved_gdf.crs)
        combined = pd.concat([dissolved_gdf, additions_gdf], ignore_index=True)
        final_dissolved = combined.dissolve(by="cluster").reset_index()

        return final_dissolved
 
    def _split_building_blocks(self, points):
        """
        Run the full pipeline for splitting all polygons whose population exceeds the threshold.
        """

        to_split = self.data[self.data["pop"] > self.pop_max].copy()
        if len(to_split) == 0:
            return self.data
        
        print(f"\nNumber of Building Blocks to split: {len(to_split)}")

        results = []
        for building_block_id in to_split[self.poly_id].unique():
            print(f"Processing Building Block: {building_block_id}")
            results.append(self._split_single_building_block(points, building_block_id))
        
        split_polys = pd.concat(results, ignore_index=True)

        # Identify overlapping or partially hidden geometries in the result
        clipped_geoms, overlap_indices = self.hidden_processor.find_partial_overlaps(split_polys, 0.005)
        hidden_gdf = split_polys[split_polys.index.isin(overlap_indices)]
        print("N° of overlapped polygons:",len(hidden_gdf))
        
        if not hidden_gdf.empty:
            split_polys.loc[clipped_geoms.index, 'geometry'] = clipped_geoms.geometry

        # Check and report multipart geometries
        split_polys, duplicates = self.identify_multipart_polygons(split_polys, self.poly_id)
        if not duplicates.empty:
            print(f"Warning: {len(duplicates)} multipart polygons still remain after processing.")
        else:
            print("No multipart polygons found.")

        remaining_polys = self.data[~self.data[self.poly_id].isin(to_split[self.poly_id])].copy()
        remaining_polys = remaining_polys.drop(columns="pop")
        
        self.data = pd.concat([remaining_polys, split_polys], ignore_index=True).copy()
        self.data[self.poly_id] = self.data[self.poly_id].str.ljust(18, fillchar='0')

        return self.data
import geopandas as gpd
import pandas as pd
import fiona
from functools import reduce
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.prepared import prep
import numpy as np
from shapely.vectorized import contains

class PolygonProcessor():
    """
    Abstract base class for polygon processing operations.
    """

    def __init__(self, data=None, poly_id=None):
        self.data = data
        self.poly_id = poly_id

    def _validate_crs(self, gdf, target_crs=32719):
        """Ensure GeoDataFrame is in target CRS."""
        if gdf.crs != target_crs:
            return gdf.to_crs(target_crs)
        return gdf

    @staticmethod
    def identify_multipart_polygons(gdf, poly_id, keep_largest=True):
        """
        Identifies and optionally resolves multipart polygons in a GeoDataFrame.

        Args:
            gdf (GeoDataFrame): Input GeoDataFrame with polygons.
            poly_id (str): Column name used to identify duplicates.
            keep_largest (bool): If True, retains only the largest polygon per ID.

        Returns:
            Tuple[GeoDataFrame, GeoDataFrame]: 
                - Cleaned GeoDataFrame (optionally keeping only the largest part).
                - GeoDataFrame with multipart polygon parts (empty if none).
        """
        # Explode to convert multipart polygons into single parts
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)

        # Identify features with multiple parts (same ID appears more than once)
        multipart_parts = gdf[gdf.duplicated(poly_id, keep=False)]

        if not multipart_parts.empty:
            if keep_largest:
                gdf['area'] = gdf.geometry.area
                gdf = gdf.loc[gdf.groupby(poly_id)['area'].idxmax()]
                gdf = gdf.drop(columns=['area'])
            return gdf, multipart_parts

        return gdf, gpd.GeoDataFrame(columns=gdf.columns)

    def combine_layers_from_gpkg(self, gpkg_path, layer_names=[13111, 13110, 13112, 13202, 13201, 13131, 13203]):
        """
        Combines selected layers from a GeoPackage into one GeoDataFrame.

        Args:
            gpkg_path (Path or str): Path to the GeoPackage.
            layer_names (list): List of layer names or IDs to combine.

        Returns:
            GeoDataFrame: Combined GeoDataFrame.
        """
        layers = fiona.listlayers(gpkg_path)

        # Convert layer names to strings for matching
        layer_names = [str(layer) for layer in layer_names]
        filtered_layers = [x for x in layers if x in layer_names]

        combined_gdf = gpd.GeoDataFrame()

        for layer in filtered_layers:
            gdf = gpd.read_file(gpkg_path, layer=layer)
            combined_gdf = pd.concat([combined_gdf, gdf], ignore_index=True)

        combined_gdf = self._validate_crs(combined_gdf)

        return combined_gdf

    def fill_holes(self, geom, sizelim=5.0):
        """
        Removes interior holes in Polygon or MultiPolygon geometries smaller than a given area threshold.

        Args:
            geom (Polygon or MultiPolygon): Input geometry.
            sizelim (float): Maximum area of holes to fill.

        Returns:
            Polygon or MultiPolygon: Geometry with small holes filled.
        """
        if geom.is_empty:
            return geom

        if geom.geom_type == 'Polygon':
            if geom.interiors:
                small_holes = [Polygon(ring) for ring in geom.interiors if Polygon(ring).area < sizelim]
                if small_holes:
                    # Union the polygon with its small holes to "fill" them
                    return reduce(lambda g1, g2: g1.union(g2), [geom] + small_holes)
            return geom

        elif geom.geom_type == 'MultiPolygon':
            processed_polygons = [self.fill_holes(polygon, sizelim) for polygon in geom.geoms]
            return MultiPolygon(processed_polygons)

        else:
            return geom

    def resolve_multipart_polygons(self, gdf, region, verbose=True):
        """
        Repairs gaps from multipart polygons by reallocating smaller orphaned fragments 
        to adjacent polygons, ensuring gap-free and topologically consistent tessellation.

        Args:
            gdf (GeoDataFrame): Input tessellated polygons.
            region (GeoDataFrame): Single polygon GeoDataFrame defining clipping boundary.
            verbose (bool): Print processing warnings and status.

        Returns:
            GeoDataFrame: Cleaned GeoDataFrame with resolved multipart polygons.
        """
        # Identify multipart polygons, keep only largest parts
        voronoi, duplicates = self.identify_multipart_polygons(gdf.copy(), self.poly_id, keep_largest=True)

        # Dissolve discarded fragments by polygon ID
        remaining_duplicates = duplicates[~duplicates.index.isin(voronoi.index)]
        combined_polys = remaining_duplicates.dissolve(by=self.poly_id).reset_index()

        for _, row in combined_polys.iterrows():
            block_id = row[self.poly_id]
            poly = row['geometry']

            # Find neighbours touching the orphan polygon but with different IDs
            neighbors = voronoi[(voronoi[self.poly_id] != block_id) & (voronoi.touches(poly))].copy()

            if not neighbors.empty:
                # Choose neighbour sharing the longest boundary
                neighbors['shared_length'] = neighbors.geometry.apply(lambda x: poly.intersection(x).length)
                best_neighbor = neighbors.loc[neighbors['shared_length'].idxmax()]

                # Get neighbours of the best neighbour (excluding itself)
                neighbor_neighbors = voronoi[
                    (voronoi[self.poly_id] != best_neighbor[self.poly_id]) &
                    (voronoi.geometry.intersects(best_neighbor.geometry))
                ]

                if not neighbor_neighbors.empty:
                    # Union neighbouring polygons and fill small holes
                    continuous_area = unary_union(neighbor_neighbors.geometry.tolist())
                    continuous_area = self.fill_holes(geom=continuous_area)

                    # Add orphan polygon to best neighbour and remove overlaps with neighbours
                    updated_geom = best_neighbor.geometry.union(poly)
                    clipped_geom = updated_geom.difference(continuous_area)

                    clipped_gdf = gpd.GeoDataFrame(geometry=[clipped_geom], crs=region.crs)
                    clipped_gdf = gpd.clip(clipped_gdf, region)

                    if not clipped_gdf.empty:
                        final_geom = clipped_gdf.geometry.iloc[0]
                        if final_geom and not final_geom.is_empty:
                            voronoi.loc[best_neighbor.name, 'geometry'] = final_geom

        # Final cleanup and check for remaining multipart polygons
        gdf = voronoi.reset_index(drop=True)
        gdf, duplicates = self.identify_multipart_polygons(gdf, self.poly_id)

        if not duplicates.empty and verbose:
            print(f"Warning: {len(duplicates)} multipart polygons still remain after processing.")
        elif verbose:
            print("No multipart polygons found.")

        return gdf

    def merge_thin_areas(self, gdf, max_width=0.5):
        """
        Merges thin polygons (based on area/length ratio) into their largest intersecting neighbor.

        Args:
            gdf (GeoDataFrame): Input polygons.
            max_width (float): Threshold for max width to consider merging.

        Returns:
            GeoDataFrame: Cleaned polygons with thin areas merged.
        """
        gdf = gdf.copy().reset_index(drop=True)
        to_drop = set()

        for idx, row in gdf.iterrows():
            geom = row.geometry

            if geom is None or geom.is_empty:
                to_drop.add(idx)
                continue

            if geom.length == 0:
                to_drop.add(idx)
                continue

            # Skip polygons that are "wide" enough
            if geom.area / geom.length > max_width:
                continue

            neighbors = gdf[
                (gdf.index != idx) & 
                (~gdf.index.isin(to_drop)) & 
                (gdf.geometry.intersects(geom))
            ]

            if neighbors.empty:
                continue

            # Merge with largest neighbor by area
            best_idx = neighbors.geometry.area.idxmax()
            gdf.at[best_idx, "geometry"] = gdf.loc[best_idx].geometry.union(geom)
            to_drop.add(idx)

        return gdf.drop(index=list(to_drop)).reset_index(drop=True)

    def reassign_disconnected_parts(self, dissolved_gdf, cluster_col="cluster"):
        """
        Reassigns disconnected parts of clusters to neighboring clusters with longest shared boundaries,
        or directly to a containing cluster if fully enclosed.

        Args:
            dissolved_gdf (GeoDataFrame): Clustered Voronoi polygons.

        Returns:
            GeoDataFrame: Updated contiguous polygons per cluster.
        """

        exploded = dissolved_gdf.explode(index_parts=False).reset_index(drop=True).copy()
        exploded["area"] = exploded.geometry.area

        duplicated_mask = exploded.duplicated(cluster_col, keep=False)
        duplicated_clusters = exploded[duplicated_mask].copy()
        idx_largest = duplicated_clusters.groupby(cluster_col)["area"].idxmax()
        largest_parts = exploded.loc[idx_largest]
        non_largest_parts = duplicated_clusters.drop(index=idx_largest).copy()
        non_largest_parts["original_index"] = non_largest_parts.index

        if len(non_largest_parts) == 0:
            return dissolved_gdf
    
        # Case 1: Reassign parts intersecting only one other cluster
        contained_mask = []

        potential_containers = exploded.loc[
            ~exploded.index.isin(non_largest_parts.index)
        ].copy()
        
        for idx, row in non_largest_parts.iterrows():
            geom = row.geometry
            current_cluster = row[cluster_col]

            intersecting = potential_containers[
                (potential_containers[cluster_col] != current_cluster) &
                (potential_containers.geometry.intersects(geom))
            ]
    
            unique_clusters = intersecting[cluster_col].unique()

            if len(unique_clusters) == 1:
                new_cluster = unique_clusters[0]
                non_largest_parts.at[idx, cluster_col] = new_cluster
                contained_mask.append(idx)

        # Remove already reassigned parts from further processing
        remaining_parts = non_largest_parts[~non_largest_parts.index.isin(contained_mask)]

        unique_clusters = remaining_parts[cluster_col].unique()

        # Case 2: Two available clusters
        if len(unique_clusters) == 2:
            cluster1, cluster2 = unique_clusters
        
            # For each small part, assign to the opposite cluster
            for idx, row in remaining_parts.iterrows():
                current_cluster = row[cluster_col]
                new_cluster = cluster2 if current_cluster == cluster1 else cluster1
                remaining_parts.at[idx, cluster_col] = new_cluster

        else:
            # Case 3: General case: use longest shared boundary
            potential_neighbors = exploded[
                ~exploded[cluster_col].isin(remaining_parts[cluster_col])
            ].copy()

            for idx, row in remaining_parts.iterrows():
                current_geom = row.geometry
                current_cluster = row[cluster_col]
        
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
                    new_cluster = potential_neighbors.loc[best_neighbor_idx, cluster_col]
                    remaining_parts.at[idx, cluster_col] = new_cluster

        fixed_non_largest = pd.concat([
            non_largest_parts.loc[contained_mask],
            remaining_parts
        ])
        
        cleaned = pd.concat([
            largest_parts.drop(columns=["area"], errors="ignore"),
            fixed_non_largest.drop(columns=["area", "original_index", "centroid"], errors="ignore"),
            exploded[~exploded.index.isin(non_largest_parts.index) & 
                     ~exploded.index.isin(largest_parts.index)]
        ], ignore_index=True)

        return cleaned.dissolve(by=cluster_col).reset_index()
    
    def remove_dangles(self, poly, buffer_dist=0.01, neg_buffer_factor=5):
        """
        Remove boundary vertices that form inward 'dangles'—segments that 
        deviate into the polygon instead of following the exterior boundary.

        Args:
            poly (Polygon or MultiPolygon): Input polygon geometry.
            buffer_dist (float): Buffer distance to create expanded polygon boundary.
            neg_buffer_factor (float): Multiplier for the negative buffer used to define the inner core.

        Returns:
            Polygon or MultiPolygon: Cleaned geometry.
        """
        def _process_single_polygon(p):
            if p.is_empty or p.geom_type != "Polygon":
                return p

            # Extract original boundary coordinates
            boundary_coords = np.array(p.exterior.coords)

            # Create a small outward buffer around the polygon
            cleaned = p.buffer(buffer_dist, quad_segs=2)

            # Apply a larger inward buffer to define the inner core area
            inner_buffer = cleaned.buffer(-buffer_dist*neg_buffer_factor, quad_segs=2)
            if inner_buffer.is_empty or not inner_buffer.is_valid:
                return p

            # Prepare inner buffer for efficient spatial queries
            prepared_inner = prep(inner_buffer)

            # Mask out vertices that fall inside the inner buffer (i.e., dangles)
            mask = ~contains(prepared_inner, boundary_coords[:, 0], boundary_coords[:, 1])
            filtered_coords = boundary_coords[mask]

            # Validate polygon, fallback to original if invalid
            new_poly = Polygon(filtered_coords, holes=p.interiors)
            return new_poly if new_poly.is_valid else p

        if poly.is_empty:
            return poly

        if poly.geom_type == "Polygon":
            return _process_single_polygon(poly)

        elif poly.geom_type == "MultiPolygon":
            cleaned_parts = [_process_single_polygon(p) for p in poly.geoms]
            cleaned_multipolygon = MultiPolygon([p for p in cleaned_parts if not p.is_empty and p.is_valid])
            return cleaned_multipolygon

        else:
            return poly
        

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
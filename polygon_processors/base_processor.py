from pathlib import Path
import geopandas as gpd
import pandas as pd
import fiona
from functools import reduce
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

class PolygonProcessor():
    """
    Abstract base class for polygon processing operations.
    
    Attributes:
        root_folder (Path): Path to the root directory for data storage or outputs.
        data (Any): Placeholder for loading or storing data (e.g., a GeoDataFrame).
        poly_id (str): Name of the column used to uniquely identify polygons.
    """
    
    def __init__(self, root_folder=None, poly_id=None):
        self.root_folder = Path(root_folder) if root_folder else Path(__file__).parent
        self.data = None
        self.poly_id = poly_id

    def _validate_crs(self, gdf, target_crs=32719):
        """Ensure GeoDataFrame is in target CRS."""
        if gdf.crs != target_crs:
            return gdf.to_crs(target_crs)
        return gdf
  
    @staticmethod
    def identify_multipart_polygons(
        gdf,
        poly_id,
        keep_largest = True
    ):
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

        # Identify features that now have multiple parts (i.e., same ID appears more than once)
        multipart_parts = gdf[gdf.duplicated(poly_id, keep=False)]
    
        if not multipart_parts.empty:

            if keep_largest:
                gdf['area'] = gdf.geometry.area
                gdf = gdf.loc[gdf.groupby(poly_id)['area'].idxmax()]
                gdf = gdf.drop(columns=['area'])

            return gdf, multipart_parts

        return gdf, gpd.GeoDataFrame(columns=gdf.columns)
    
    def combine_layers_from_gpkg(
        self, 
        gpkg_path, 
        layer_names=[13111, 13110, 13112, 13202, 13201, 13131, 13203]
    ):
        """
        Combines all layers from a GeoPackage into one GeoDataFrame.
    
        Args:
            gpkg_path (Path or str): Path to the GeoPackage
    
        Returns:
            GeoDataFrame: Combined GeoDataFrame
        """
        layers = fiona.listlayers(gpkg_path)
        
        # Filter layers to combine
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
                    return reduce(lambda g1, g2: g1.union(g2), [geom] + small_holes)
            return geom
        
        elif geom.geom_type == 'MultiPolygon':
            # Process each polygon in the multipolygon
            processed_polygons = []
            for polygon in geom.geoms:
                processed = self.fill_holes(polygon, sizelim)
                processed_polygons.append(processed)
            return MultiPolygon(processed_polygons)
        
        else:
            # Return as-is for other geometry types
            return geom

    def resolve_multipart_polygons(self, gdf, region, verbose=True):
        """
        Repairs gaps from multipart polygons by reallocating smaller orphaned fragments 
        to adjacent polygons, ensuring a gap-free and topologically consistent tessellation.

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
        Merge geometries in the GeoDataFrame that are thinner than a specified width 
        by dissolving them into their largest touching neighbour.

        Args:
            gdf (GeoDataFrame): Input GeoDataFrame with polygon geometries.
            max_width (float): Maximum allowable width (area/length ratio) before merging.

        Returns:
            GeoDataFrame: Modified GeoDataFrame with thin polygons merged into neighbours.
            
        """
        gdf.reset_index(drop=True, inplace=True)

        to_drop = []

        for idx, row in gdf.iterrows():
            geom = row.geometry

            # Check if it's thin
            if geom.area / geom.length > max_width:
                continue

            # Find touching neighbours (excluding itself)
            touching = gdf[gdf.index != idx]
            touching = touching[touching.geometry.touches(geom)]

            if touching.empty:
                continue  # No valid neighbour

            # Compute shared boundary length
            touching['shared_length'] = touching.geometry.apply(
                lambda g: g.boundary.intersection(geom.boundary).length
            )

            best_idx = touching['shared_length'].idxmax()

            # Merge into the best neighbour
            merged_geom = gdf.loc[best_idx].geometry.union(geom)
            gdf.at[best_idx, 'geometry'] = merged_geom

            # Mark this thin geometry for removal
            to_drop.append(idx)

        # Drop merged thin geometries
        gdf = gdf.drop(index=to_drop).reset_index(drop=True)

        return gdf
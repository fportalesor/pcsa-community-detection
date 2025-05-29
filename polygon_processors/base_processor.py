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
        id_column (str): Name of the column used to uniquely identify polygons.
    """
    
    def __init__(self, root_folder=None, id_column=None):
        self.root_folder = Path(root_folder) if root_folder else Path(__file__).parent
        self.data = None
        self.id_column = id_column

    def _validate_crs(self, gdf, target_crs=32719):
        """Ensure GeoDataFrame is in target CRS."""
        if gdf.crs != target_crs:
            return gdf.to_crs(target_crs)
        return gdf
  
    @staticmethod
    def identify_multipart_polygons(
        gdf,
        id_column,
        keep_largest = True
    ):
        """
        Identifies and optionally resolves multipart polygons in a GeoDataFrame.

        Args:
            gdf (GeoDataFrame): Input GeoDataFrame with polygons.
            id_column (str): Column name used to identify duplicates.
            keep_largest (bool): If True, retains only the largest polygon per ID.

        Returns:
            Tuple[GeoDataFrame, GeoDataFrame]: 
                - Cleaned GeoDataFrame (optionally keeping only the largest part).
                - GeoDataFrame with multipart polygon parts (empty if none).
        """
        # Explode to convert multipart polygons into single parts
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)

        # Identify features that now have multiple parts (i.e., same ID appears more than once)
        multipart_parts = gdf[gdf.duplicated(id_column, keep=False)]
    
        if not multipart_parts.empty:

            if keep_largest:
                gdf['area'] = gdf.geometry.area
                gdf = gdf.loc[gdf.groupby(id_column)['area'].idxmax()]
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

    def repair_multipart_voronoi_gaps(self, gdf, region, buffer_dist=100):
        """
        Repairs gaps resulting from multipart polygons in a Voronoi-like tessellation
        by reallocating orphaned parts to neighbouring polygons.

        This method:
        1. Identifies multipart polygons and retains only the largest geometry for each unique ID.
        2. Dissolves the discarded parts into larger combined geometries.
        3. Finds the neighbouring polygon with the longest shared boundary.
        4. Buffers the selected neighbour and trims it to reabsorb the discarded area,
           ensuring topological consistency.
        5. Clips the updated polygon geometry to the specified region boundary using GeoPandas' clip.
        Args:
            gdf (GeoDataFrame): Input GeoDataFrame containing tessellated polygons.
            region (GeoDataFrame): GeoDataFrame with a single polygon defining the target boundary for clipping.
            buffer_dist (float, optional): Buffer distance us   ed to expand neighbouring polygons
                for reallocation. Default is 100 metres (assumes a projected coordinate system).

        Returns:
            GeoDataFrame: Cleaned and topologically consistent GeoDataFrame.
        """
        
        # Step 1: Identify multipart polygons and keep only the largest part per ID
        voronoi, duplicates = self.identify_multipart_polygons(
            gdf, self.id_column, keep_largest=True)

        # Step 2: Dissolve the discarded parts (non-largest) into single geometries per ID
        remaining_duplicates = duplicates[~duplicates.index.isin(voronoi.index)]
        combined_polys = remaining_duplicates.dissolve(by=self.id_column).reset_index()
        
        # Step 3: For each discarded geometry, assign it to the best-fitting neighbouring polygon
        for _, row in combined_polys.iterrows():
            block_id = row[self.id_column]
            poly = row['geometry']
            
            # Step 3a: Identify adjacent polygons (touching but with a different ID)
            neighbors = voronoi[voronoi[self.id_column] != block_id]
            neighbors = neighbors[neighbors.touches(poly)]
            
            if not neighbors.empty:
                # Step 3b: Compute the length of the shared boundary with each neighbour
                neighbors['shared_length'] = neighbors.geometry.apply(
                    lambda x: poly.intersection(x).length
                )
                
                # Step 3c: Choose the neighbour with the longest shared boundary
                best_neighbor = neighbors.loc[neighbors['shared_length'].idxmax()]
                
                # Step 4a: Buffer the best neighbour to ensure full coverage
                buffered = best_neighbor.geometry.buffer(buffer_dist)

                # Step 4b: Identify all polygons that intersect this buffer (i.e., neighbouring context)
                neighbor_neighbors = voronoi[
                    (voronoi[self.id_column] != best_neighbor[self.id_column])
                ]
                neighbor_neighbors = neighbor_neighbors[neighbor_neighbors.intersects(buffered)]
                               
                if not neighbor_neighbors.empty:
                    # Step 4c: Build a unified geometry from the surrounding neighbours
                    continuous_area = unary_union(neighbor_neighbors.geometry.tolist())

                    # Step 4d: Fill small holes in the unioned area (optional cleanup)
                    continuous_area = self.fill_holes(geom=continuous_area)
                    
                    # Step 4e: Trim the buffered polygon to avoid overlapping into neighbour areas
                    clipped_geom = buffered.difference(continuous_area)
                    clipped_gdf = gpd.GeoDataFrame(geometry=[clipped_geom], crs=region.crs)

                    # Step 4f: Clip the trimmed geometry with the region boundary
                    clipped_gdf = gpd.clip(clipped_gdf, region)

                    # Extract the clipped geometry
                    if not clipped_gdf.empty:
                        final_geom = clipped_gdf.geometry.iloc[0]
                    else:
                        final_geom = None

                    # Step 4e: Update the geometry of the best neighbour with the trimmed buffer
                    if final_geom and not final_geom.is_empty:
                        voronoi.loc[best_neighbor.name, 'geometry'] = final_geom
        
        # Step 5: Final cleanup – reset index and check for any remaining multipart polygons
        gdf = voronoi.reset_index(drop=True)
        gdf, duplicates = self.identify_multipart_polygons(gdf, self.id_column)

        if not duplicates.empty:
            print(f"Warning: {len(duplicates)} multipart polygons still remain after processing.")

        else:
            print("No multipart polygons found.")

        return gdf
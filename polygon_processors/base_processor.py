from pathlib import Path
import geopandas as gpd
from abc import ABC
from functools import reduce
from shapely.geometry import Polygon
from shapely.ops import unary_union

class PolygonProcessor(ABC):
    """Abstract base class for polygon processing operations."""
    
    def __init__(self, root_folder=None):
        self.root_folder = Path(root_folder) if root_folder else Path(__file__).parent

        self.data = None
    
    def concat_polys(self, folder_name="processed_data"):
        """
        Load and concatenate all Voronoi shapefiles (*voronoi*.shp) from specified folder.
        
        Args:
            folder_name (str): Subfolder containing Voronoi shapefiles. Defaults to "processed_data".
            
        Returns:
            GeoDataFrame: Combined Voronoi polygons in target CRS.
        """
        # Find all voronoi shapefiles in folder
        voronoi_files = list((self.root_folder / folder_name).glob("voronoi_*.shp"))
        
        if not voronoi_files:
            raise FileNotFoundError(f"No Voronoi shapefiles found in {folder_name}")
            
        # Read and concatenate all files
        gdf_list = [gpd.read_file(f) for f in voronoi_files]
        combined = gpd.GeoDataFrame(gpd.pd.concat(gdf_list, ignore_index=True))
        
        # Validate CRS and store in instance
        self.data = self._validate_crs(combined)
        return self.data
    
    def _validate_crs(self, gdf, target_crs=32719):
        """Ensure GeoDataFrame is in target CRS."""
        if gdf.crs != target_crs:
            return gdf.to_crs(target_crs)
        return gdf
    
    def fill_holes(self, geom, sizelim):
        """Fill holes in a polygon that are smaller than the area threshold."""
        if geom.interiors:
            small_holes = [Polygon(ring) for ring in geom.interiors if Polygon(ring).area < sizelim]
            if small_holes:
                return reduce(lambda g1, g2: g1.union(g2), [geom] + small_holes)
        return geom

    def process_duplicate_voronois(self, gdf):
        """
        Process duplicate MANZENT polygons by:
        1. Keeping the largest area for each duplicate group
        2. Combining remaining duplicates into single polygons
        3. Finding neighboring polygons and modifying them
        """
        if gdf is None:
            raise ValueError("No data loaded. Run concat_polys() first.")
            
        # Step 1: Explode multipolygons
        voronoi = gdf.explode(index_parts=False).reset_index(drop=True)
        
        # Step 2: Identify duplicates
        duplicates = voronoi[voronoi.duplicated('MANZENT', keep=False)]
        
        if duplicates.empty:
            print("No duplicates found")
            gdf = voronoi
            return gdf
            
        # Step 3: For each duplicate group, keep the largest area
        voronoi['area'] = voronoi.geometry.area
        voronoi = voronoi.loc[voronoi.groupby("MANZENT")['area'].idxmax()]
        voronoi = voronoi.drop(columns=['area'])
        
        # Step 4: Combine remaining duplicates into single polygons
        remaining_duplicates = duplicates[~duplicates.index.isin(voronoi.index)]
        combined_polys = remaining_duplicates.dissolve(by='MANZENT').reset_index()
        
        #combined_polys.to_file("comb_rem_polys.shp")
        # Step 5: For each combined polygon, find neighbor with longest shared boundary
        for _, row in combined_polys.iterrows():
            manzent = row['MANZENT']
            poly = row['geometry']
            
            # Find all neighboring polygons (different MANZENT)
            neighbors = voronoi[voronoi['MANZENT'] != manzent]
            neighbors = neighbors[neighbors.touches(poly)]
            
            if not neighbors.empty:
                # Calculate shared boundary length with each neighbor
                neighbors['shared_length'] = neighbors.geometry.apply(
                    lambda x: poly.intersection(x).length
                )
                
                # Get neighbor with longest shared boundary
                best_neighbor = neighbors.loc[neighbors['shared_length'].idxmax()]
                
                # Step 6: Buffer the neighbor and find its neighbors
                buffered = best_neighbor.geometry.buffer(100)

                #buffered_gdf = gpd.GeoDataFrame(geometry=[buffered], crs=voronoi.crs)
                #buffered_gdf.to_file("buffered_gdf.shp")

                neighbor_neighbors = voronoi[
                    (voronoi['MANZENT'] != best_neighbor['MANZENT'])
                ]
                neighbor_neighbors = neighbor_neighbors[neighbor_neighbors.intersects(buffered)]
                
                # Step 7: Create continuous area with neighbor's neighbors
                if not neighbor_neighbors.empty:
                    continuous_area = unary_union(neighbor_neighbors.geometry.tolist())
                    continuous_area = self.fill_holes(geom=continuous_area, sizelim=5.0)

                    #continuous_area_gdf = gpd.GeoDataFrame(geometry=[continuous_area], crs=voronoi.crs)
                    #continuous_area_gdf.to_file("continuous_area_gdf.shp")
                    
                    # Step 8: Apply intersection between buffered neighbor and continuous area
                    clipped_geom = buffered.difference(continuous_area)

                    #clipped_gdf = gpd.GeoDataFrame(geometry=[clipped_geom], crs=voronoi.crs)
                    #clipped_gdf.to_file("new_geom_gdf.shp")

                    # Update the neighbor's geometry
                    voronoi.loc[best_neighbor.name, 'geometry'] = clipped_geom
                           
        gdf = voronoi.reset_index(drop=True)
        return gdf
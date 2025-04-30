import geopandas as gpd
import topojson as tp
from longsgis import voronoiDiagram4plg
from .base_processor import PolygonProcessor
from .densifier import PolygonDensifier
from .hidden_polys import HiddenPolygonProcessor

class VoronoiProcessor(PolygonProcessor):
    """Creates and processes Voronoi diagrams from input polygons."""
    
    def __init__(self, input_data=None, region_id=None, root_folder=None):
        super().__init__(root_folder)
        self.data = input_data
        self.region_id = region_id
        self.densifier = PolygonDensifier()
        self.hidden_processor = HiddenPolygonProcessor()
    
    def process(self, bbs_path=None, region_path="data/COMUNA_C17.shp", 
                densify_distance=5, buffer_filler=10, buffer_reduction=-5.5,
                buffer_region=10, tolerance=1, overlay_hidden=False):
        """
        Create and process Voronoi polygons.
        
        Args:
            bbs_path (str): Path to input polygons (optional if data loaded)
            region_path (str): Path to region boundary data
            densify_distance (float): Distance for vertex densification
            buffer_filler (float): Buffer distance for filling polygon gaps
            buffer_reduction (float): Buffer negative distance to reduce the area
            
        Returns:
            GeoDataFrame: Processed Voronoi polygons
        """
        if self.data is None and bbs_path:
            self.data = gpd.read_file(bbs_path)
        
        self._prepare_input_polygons(buffer_filler, buffer_reduction)
        region = self._load_and_filter_region(region_path)
        self._filter_polygons_in_region(region)
        self._densify_polygons(densify_distance)
        voronoi = self._create_voronoi_diagram(region, buffer_region)
        voronoi = self._process_hidden_polygons(voronoi, apply_overlay=overlay_hidden)
        voronoi = self._simplify_boundaries(voronoi, region, tolerance)

        print("N° of resulting polygons:",len(voronoi), "\n")
        
        self.data = voronoi
        return self.data
    
    def _prepare_input_polygons(self, buffer_filler, buffer_reduction):
        """Prepare input polygons by filling small gaps, smoothing edges,
           and applying a final buffer reduction."""
        
        self.data = self._validate_crs(self.data)
        self.data = self.data.assign(geometry=self.data.geometry.buffer(buffer_filler))
        self.data = self.data.assign(geometry=self.data.geometry.buffer(-buffer_filler))

        self.data = self.data.assign(geometry=self.data.geometry.buffer(buffer_reduction))
        self.data = self.data.explode(index_parts=False).reset_index(drop=True)
        
        # Some polygons may have segments narrower than the buffer_reduction distance, causing fragmentation.
        # To mitigate this, only the largest polygon is retained.
        self.data['area'] = self.data.geometry.area
        self.data = self.data.loc[self.data.groupby("MANZENT")['area'].idxmax()]
        self.data = self.data.drop(columns=['area'])
    
    def _load_and_filter_region(self, region_path):
        """Load and filter region boundary data."""
        region = gpd.read_file(region_path)
        region = self._validate_crs(region)
        region["COMUNA"] = region["COMUNA"].astype(int)
        return region.loc[region["COMUNA"] == self.region_id]
    
    def _filter_polygons_in_region(self, region):
        """Filter polygons to only those within the target region."""
        centroids = self.data.copy()
        centroids['geometry'] = centroids['geometry'].apply(lambda geom: geom.representative_point())
        centroids = centroids.sjoin(region, how="inner", predicate="intersects")
        self.data = self.data.merge(centroids[['MANZENT']], on="MANZENT", how="inner")

        print("REGION CODE:", self.region_id, " / ",
              "NAME:", region.NOM_COMUNA.iloc[0])
        print("N° of original polygons:",len(self.data))
    
    def _densify_polygons(self, distance):
        """Densify polygon vertices for better Voronoi results."""
        self.data = self.densifier.densify_geodataframe(self.data, distance)
    
    def _create_voronoi_diagram(self, region, buffer_region=10):
        """Create constrained Voronoi diagram."""
        # Apply a buffer to the simplified geometriesregion geometry to mitigate gaps introduced during simplification,
        # ensuring alignment with the original region boundaries after clipping
        region = region.assign(geometry=region.geometry.buffer(buffer_region))
        voronoi = voronoiDiagram4plg(self.data, region)
        return voronoi.explode(index_parts=False).reset_index(drop=True)
    
    def _process_hidden_polygons(self, voronoi, apply_overlay=False):
        """Handle hidden/overlapped polygons in Voronoi diagram.
    
         Args:
            voronoi (GeoDataFrame): The Voronoi diagram to process
            apply_overlay (bool): If True, will perform an overlay between visible and hidden polygons
        
        Returns:
            GeoDataFrame: Processed Voronoi polygons
        """
        _, hidden_indices = self.hidden_processor.find_hidden_polygons(voronoi)
        hidden_gdf = voronoi[voronoi.index.isin(hidden_indices)]
        print("N° of hidden polygons:",len(hidden_gdf))
        
        if not hidden_gdf.empty:
            hidden_gdf.to_file(self.root_folder / f"processed_data/hidden_{self.region_id}.shp")
            visible = voronoi[~voronoi.index.isin(hidden_indices)]

            if apply_overlay:
                # Perform overlay operation between visible and hidden polygons
                union = gpd.overlay(
                    visible,
                    hidden_gdf,
                    how='union'
                )

                # Fill NaN values from hidden_gdf (_2) with visible (_1) values
                for col in visible.columns:
                    if col != 'geometry':  # Skip geometry column
                        col_1 = f"{col}_1"
                        col_2 = f"{col}_2"
                    
                        if col_1 in union.columns and col_2 in union.columns:
                            # Fill NaN _2 values with _1 values
                            union[col_2] = union[col_2].fillna(union[col_1])
                            # Keep only the _2 column (now with filled values)
                            union[col] = union[col_2]
                            union.drop(columns=[col_1, col_2], inplace=True)
                            union["CUT"] = union["CUT"].astype(int)

                return union
            return visible
        return voronoi
    
    def _simplify_boundaries(self, gdf, region_gdf, tolerance=1):
        """Simplifies polygon boundaries and clips to a region.
    
        Uses topological simplification to reduce vertex count while preserving structure,
        then clips results to the specified region boundaries.

        Args:
            gdf (GeoDataFrame): Input polygons to simplify
            region_gdf (GeoDataFrame): Boundary for final clipping
            tolerance (float): Simplification tolerance (metres)

        Returns:
            GeoDataFrame: Simplified and clipped polygons
        """
        topo = tp.Topology(gdf, prequantize=True)
        simplified = topo.toposimplify(tolerance).to_gdf()
    
        clipped = gpd.clip(simplified, region_gdf)
        return clipped
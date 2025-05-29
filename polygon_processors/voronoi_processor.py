import geopandas as gpd
import topojson as tp
from longsgis import voronoiDiagram4plg
from .base_processor import PolygonProcessor
from .densifier import PolygonDensifier
from .hidden_polys import HiddenPolygonProcessor

class VoronoiProcessor(PolygonProcessor):
    """
    Class to create and process Voronoi diagrams from input polygon data.

    This processor takes polygon input (GeoDataFrame or file), applies buffering, densification,
    Voronoi diagram creation, handling of hidden polygons, and boundary simplification,
    all constrained within a specified region.

    Attributes:
        data (GeoDataFrame or None): Input polygons to process.
        id_column (str or None): Column name for polygon IDs.
        region_id (int or None): Identifier for the target region to process.
        subregion_col (str or None): Column name for subregion IDs.
        root_folder (Path or str or None): Directory for saving intermediate files.
        densifier (PolygonDensifier): Helper class to densify polygon vertices.
        hidden_processor (HiddenPolygonProcessor): Helper class to identify hidden polygons.
    """
    
    def __init__(self, input_data=None, id_column="block_id", region_id=None, 
                 subregion_col="subregion_id", root_folder=None):
        
        self.data = input_data
        self.id_column = id_column
        self.region_id = region_id
        self.subregion_col = subregion_col
        self.root_folder = root_folder
        self.densifier = PolygonDensifier()
        self.hidden_processor = HiddenPolygonProcessor()
    
    def process(self, bbs_path=None, region_path="data/raw/COMUNA_C17.shp", 
                barrier_mask_path="data/raw/hidrographic_network.shp",
                barrier_buffer=1.0,
                densify_distance=5.0, buffer_filler=10.0, buffer_reduction=5.5,
                buffer_region=10.0, tolerance=1.0, overlay_hidden=False):
        """
        Create and process Voronoi polygons.
        
        Args:
            bbs_path (str): Path to input polygons (optional if data loaded).
            region_path (str): Path to region boundary data.
            barrier_buffer (float): Buffer distance (in CRS units) to expand the barrier
                geometries before subtraction. Default is 1 unit.
            densify_distance (float): Distance for vertex densification.
            buffer_filler (float): Buffer distance for filling polygon gaps.
            buffer_reduction (float): Buffer negative distance to reduce the area.
            buffer_region (float): Buffer distance in metres applied to the region boundary for Voronoi clipping.
            tolerance (float): Simplification tolerance in metres for polygon boundaries.
            overlay_hidden (bool): If True, overlays visible and hidden polygons to merge them.
            
        Returns:
            GeoDataFrame: Processed Voronoi polygons
        """
        if self.data is None and bbs_path:
            self.data = gpd.read_file(bbs_path)
        
        region, subregion = self._prepare_region(region_path, barrier_mask_path, barrier_buffer)
        self._filter_polygons_in_region(region)

        n_orig_polys = len(self.data)
        print("N° of original polygons:", n_orig_polys )

        self._prepare_input_polygons(buffer_filler, buffer_reduction, subregion, self.subregion_col)
        self._densify_polygons(densify_distance)
        voronoi = self._create_voronoi_diagram(region, buffer_region)
        voronoi = self._process_hidden_polygons(voronoi, apply_overlay=overlay_hidden)
        voronoi = self._simplify_boundaries(voronoi, region, tolerance)
        voronoi = self.repair_multipart_voronoi_gaps(gdf=voronoi, region=region)

        n_final_polys = len(voronoi)
        print("N° of resulting polygons:", n_final_polys)

        print("-----------------------------------------------")
        self.data = voronoi
        return self.data
    
    def _prepare_input_polygons(self, buffer_filler, buffer_reduction, subregion_gdf, subregion_id):
        """Clean and prepare polygon geometries by smoothing edges, reducing size, 
        and assigning subregion IDs.
        
        Args:
            buffer_filler (float): Buffer size to fill small gaps and smooth geometry.
            buffer_reduction (float): Buffer size to reduce polygon size after smoothing.
            subregion_gdf (GeoDataFrame): Subregion boundaries.
            subregion_id (str): Column in `subregion_gdf` for subregion assignment.
        """
        
        self.data = self._validate_crs(self.data)

        self.data['geometry'] = self.data.geometry.buffer(buffer_filler)
        self.data['geometry'] = self.data.geometry.buffer(-buffer_filler)
        self.data['geometry'] = self.data.geometry.buffer(-buffer_reduction)
 
        # Some polygons may have segments narrower than the buffer_reduction distance, causing fragmentation.
        # To mitigate this, only the largest polygon is retained.
        self.data, _ = self.identify_multipart_polygons(self.data, self.id_column, keep_largest=True)

        self._assign_subregion_ids(subregion_gdf, subregion_id)


    def _prepare_region(self, region_path, barrier_mask_path, barrier_buffer):
        """
        Load and filter region boundary data, applying optional barrier splitting.
    
        Args:
            region_path (str): Path to the shapefile defining the target region boundaries.
            barrier_mask_path (str or None): Path to a barrier mask shapefile (e.g., hydrographic
                network) used to split the region into subregions based on physical barriers
                (like rivers or lakes). This ensures Voronoi polygons do not extend across
                these natural divisions.
            barrier_buffer (float): Buffer distance (in CRS units) to expand the barrier
                geometries before subtraction. Default is 1 unit.

        Returns:
            Tple[GeoDataFrame, GeoDataFrame or None]: 
                - Original region (GeoDataFrame).
                - Subregions (GeoDataFrame) if barrier lines were provided; otherwise, None.
        """
        region = gpd.read_file(region_path)
        region = region.rename(columns={"COMUNA": "commune_id",
                                        "NOM_COMUNA": "commune"})
                
        region = self._validate_crs(region)
        region["commune_id"] = region["commune_id"].astype(int)
        region = region.loc[region["commune_id"] == self.region_id]

        if barrier_mask_path:
            barrier_mask = gpd.read_file(barrier_mask_path)
            barrier_mask = self._validate_crs(barrier_mask)
            barrier_mask["geometry"] = barrier_mask.geometry.buffer(barrier_buffer)

            # Difference operation
            subregions = gpd.overlay(region, barrier_mask, how='difference')

            # If no intersection with the barrier occurs, fallback to original region.
            # This ensures that at least one subregion is created even when the barrier 
            # does not pass through the target region.

            if not subregions.empty:
                subregions = subregions.explode(index_parts=False).reset_index(drop=True)
            else:
                subregions = region.copy()

            # Create subregion identifiers
            subregions['count'] = subregions.groupby("commune_id").cumcount() + 1
            subregions = subregions.reset_index(drop=True)
            subregions['count'] = subregions['count'].astype(str).str.zfill(2)
            subregions[self.subregion_col] = subregions["commune_id"].astype(str) + subregions["count"]

            return region, subregions

        return region, None
    
    def _filter_polygons_in_region(self, region):
        """Filter polygons to only those within the target region."""
        centroids = self.data.copy()
        centroids['geometry'] = centroids['geometry'].apply(lambda geom: geom.representative_point())
        centroids = centroids.sjoin(region, how="inner", predicate="intersects")
        self.data = self.data.merge(centroids[[self.id_column]], on=self.id_column, how="inner")

        print("-----------------------------------------------")
        print("Region Code:", self.region_id, " - ",
              "Name:", region.commune.iloc[0], "\n")
        
    def _assign_subregion_ids(self, subregions, subregion_id):
        """
        Assigns subregion identifiers to each Voronoi polygon based on its centroid's location.

        Args:
            subregions (GeoDataFrame): GeoDataFrame containing the subregion geometries and subregion IDs.
            subregion_id (str): Column name in `subregions` containing the subregion identifier.

        Modifies:
            self.data (GeoDataFrame): Adds a column with subregion IDs assigned to each feature.
        """
        centroids = self.data.copy()
        centroids['geometry'] = centroids['geometry'].apply(lambda geom: geom.representative_point())

        # Perform spatial join with subregions to get subregion ID per centroid
        centroids = centroids.sjoin(
            subregions[["geometry", subregion_id]],
            how="left",
            predicate="intersects")

        # Merge the subregion ID back into the original self.data
        self.data = self.data.merge(
            centroids[[self.id_column, subregion_id]],
            on=self.id_column, 
            how="left")
        
    def _densify_polygons(self, distance):
        """Densify polygon vertices for better Voronoi results."""
        self.data = self.densifier.densify_geodataframe(self.data, distance)
    
    def _create_voronoi_diagram(self, region, buffer_region=10):
        """Create constrained Voronoi diagram."""
        # Apply a buffer to the simplified geometries region geometry to mitigate gaps introduced during simplification,
        # ensuring alignment with the original region boundaries after clipping
        region_buffered = region.copy()
        region_buffered["geometry"] = region_buffered.geometry.buffer(buffer_region)
        voronoi = voronoiDiagram4plg(self.data, region_buffered)
        return voronoi
    
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
            gpkg_path = self.root_folder / "hidden_polys.gpkg"
            layer_name = str(self.region_id)

            # Write the new layer
            hidden_gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG", mode="w")

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
                            union["commune_id"] = union["commune_id"].astype(int)

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
import geopandas as gpd
import pandas as pd
import topojson as tp
from longsgis import voronoiDiagram4plg
from .base_processor import PolygonProcessor
from .densifier import PolygonDensifier
from .hidden_polys import HiddenPolygonProcessor
from .parallel_voronoi import ParallelVoronoiProcessor

class VoronoiProcessor(PolygonProcessor):
    """
    Class to create and process Voronoi diagrams from input polygon data.

    This processor takes polygon input (GeoDataFrame or file), applies buffering, densification,
    Voronoi diagram creation, handling of hidden polygons, and boundary simplification,
    all constrained within a specified region.
    """
    
    def __init__(self, input_data=None, poly_id="block_id", region_id=None, 
                 subregion_id="sregion_id"):
        
        self.data = input_data
        self.poly_id = poly_id
        self.region_id = region_id
        self.subregion_id = subregion_id
        self.densifier = PolygonDensifier()
        self.hidden_processor = HiddenPolygonProcessor()
        """
        Args:
            data (GeoDataFrame or None): Input polygons to process.
            poly_id (str or None): Column name for polygon IDs.
            region_id (int or None): Identifier for the target region to process.
            subregion_id (str or None): Column name for subregion IDs.
            densifier (PolygonDensifier): Helper class to densify polygon vertices.
            hidden_processor (HiddenPolygonProcessor): Helper class to identify hidden polygons.
        """
        
    def process(self, bbs_path=None, 
                region_path="data/raw/COMUNA_C17.shp", 
                barrier_mask_path="data/raw/hidrographic_network.shp",
                barrier_buffer=1.0,
                int_region_path="data/raw/ZONA_C17.shp",
                clean_polys=True, buffer_filler=10.0, buffer_reduction=5.5,
                densify_bdry=True, densify_distance=5.0,
                buffer_region=10.0, 
                simplify_bdry=True, tolerance=1.0,
                process_hidden=True,
                overlay_hidden=False,
                return_hidden=False,
                fix_multipart=True,
                verbose=True,
                by_chunks=True):
        """
        Create and process Voronoi polygons from input data with optional parallel processing.
        
        Args:
            bbs_path (str): Path to input polygons (optional if already loaded).
            region_path (str): Path to region boundary shapefile.
            barrier_mask_path (str): Path to barrier features (e.g., rivers).
            barrier_buffer (float): Buffer distance for barrier features before subtraction.
            int_region_path (str, optional): Path to intermediate region splitting boundaries.
                Used when by_chunks=True.
            clean_polys (bool): Whether to clean input polygons before processing.
            buffer_filler (float): Buffer distance to close gaps in input geometries.
            buffer_reduction (float): Negative buffer to shrink polygons after filling gaps.
            densify_bdry (bool): Whether to densify polygon boundaries before Voronoi generation.
            densify_distance (float): Spacing of points added during boundary densification.
            buffer_region (float): Buffer around region boundary used to clip Voronoi.
            simplify_bdry (bool): Whether to simplify polygon geometries at the end.
            tolerance (float): Tolerance (in units of CRS) used for simplification.
            process_hidden (bool): Whether to identify and exclude or merge hidden Voronoi polygons.
            overlay_hidden (bool): If True and process_hidden is True, merge hidden and visible polygons using overlay.
            return_hidden (bool): If True and process_hidden is True, also return GeoDataFrame of hidden polygons.
            fix_multipart (bool, optional): Whether to repair gaps caused by multipart polygons.
                If True, small disconnected parts will be merged with neighbouring polygons.
                Defaults to True.
            verbose (bool): If True, prints progress messages and intermediate information during processing.
            by_chunks (bool): If True, processes region in parallel chunks. Defaults to True.
            
        Returns:
            GeoDataFrame:
                Processed Voronoi polygons, if `return_hidden` is False.

            tuple[GeoDataFrame, GeoDataFrame]:
                A tuple (processed Voronoi polygons, hidden polygons), if `return_hidden` is True.
        """
        if self.data is None:
            if not bbs_path:
                raise ValueError("No input data provided: self.data is None and bbs_path is not set.")
            
            self.data = gpd.read_file(bbs_path)
    
        region, subregion = self._prepare_region(region_path, barrier_mask_path, barrier_buffer)
        self.data = self._filter_polygons_in_region(self.data, region, verbose)

        if verbose:
            print(f"No. of original polygons: {len(self.data)}")

        if clean_polys:
            self._prepare_input_polygons(buffer_filler, buffer_reduction, subregion, self.subregion_id)
        if densify_bdry:
            self._densify_polygons(densify_distance)

        # Process either by chunks or as a whole
        if by_chunks:
            region_splitter = self._prepare_intermediate_region(int_region_path)
            region_chunks = self._create_region_chunks(region, region_splitter, verbose)
        
            # Process chunks in parallel
            voronoi, all_hidden = ParallelVoronoiProcessor(
                n_jobs=-1, 
                verbose=verbose
            ).parallel_process(
                voronoi_processor=self,
                region_chunks=region_chunks,
                region=region,
                buffer_region=buffer_region,
                process_hidden=process_hidden,
                overlay_hidden=overlay_hidden,
                simplify_bdry=simplify_bdry,
                tolerance=tolerance,
                verbose=verbose
            )
            if verbose and all_hidden is not None and not all_hidden.empty:
                print("No. of hidden polygons:", len(all_hidden))

            if fix_multipart:
                voronoi = self.resolve_multipart_polygons(voronoi, region, verbose)
        else:
            # Process whole region at once
            voronoi = self._create_voronoi_diagram(self.data, region, buffer_region)
        
            hidden_gdf = None
            if process_hidden:
                voronoi, hidden_gdf = self._process_hidden_polygons(voronoi, overlay_hidden, verbose)
            if simplify_bdry:
                voronoi = self._simplify_boundaries(voronoi, region, tolerance)
            if fix_multipart:
                voronoi = self.resolve_multipart_polygons(voronoi, region, verbose)
        
            all_hidden = hidden_gdf

        if verbose:
            print(f"No. of resulting polygons: {len(voronoi)}\n-----------------------------------------------")
        
        return (voronoi, all_hidden) if (process_hidden and return_hidden) else voronoi
    
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
        self.data, _ = self.identify_multipart_polygons(self.data, self.poly_id, keep_largest=True)

        self.data = self._assign_subregion_ids(subregion_gdf, subregion_id)
        return self.data

    def _prepare_region(self, region_path, barrier_mask_path, barrier_buffer):
        """
        Load and filter region boundary data, applying optional barrier splitting.
    
        Args:
            region_path (str): Path to the shapefile defining the target region boundaries.
            barrier_mask_path (str or None): Path to a barrier mask shapefile (e.g., hydrographic
                network) used to split the region into subregions based on physical barriers
                (like rivers or lakes). This ensures that subsequent aggregation of Voronoi polygons is 
                constrained within these natural divisions.
            barrier_buffer (float): Buffer distance (in CRS units) to expand the barrier
                geometries before splitting. Default is 1 unit.

        Returns:
            Tuple[GeoDataFrame, GeoDataFrame or None]: 
                - Original region (GeoDataFrame).
                - Subregions (GeoDataFrame) if barrier lines were provided; otherwise, None.
        """
        region = gpd.read_file(region_path)
        
        # Rename columns to standard names (specific to Chilean Census dataset)
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
            subregions[self.subregion_id] = subregions["commune_id"].astype(str) + subregions["count"]

            return region, subregions

        return region, None
    
    def _prepare_intermediate_region(self, int_region_path):
        """
        Loads and prepares intermediate region geometries for a specified region.

        This function reads a GeoDataFrame from the given path, validates the CRS,
        renames relevant columns and filters by the current region ID.

        Args:
            int_region_path (str): Path to the spatial file containing intermediate region geometries.

        Returns:
            geopandas.GeoDataFrame: A filtered and formatted GeoDataFrame of intermediate regions.
        """
        int_region = gpd.read_file(int_region_path)
        int_region = self._validate_crs(int_region)

        int_region = int_region.rename(columns={"COMUNA": "commune_id",
                                                "NOM_COMUNA": "commune"})
        
        int_region["commune_id"] = int_region["commune_id"].astype(int)
        int_region = int_region.loc[int_region["commune_id"] == self.region_id]

        return int_region

    def _create_region_chunks(self, region, int_region, verbose=False):
        """
        Splits a region into chunks using intermediate regions and handles boundary mismatches.

        Args:
            region (geopandas.GeoDataFrame): The full region geometry.
            int_region (geopandas.GeoDataFrame): Intermediate regions used to split the main region.
            verbose (bool, optional): If True, prints the number of resulting chunks. Defaults to False.

        Returns:
            geopandas.GeoDataFrame: A cleaned GeoDataFrame of region chunks.
        """
        # Split region in chunks using intermediate regions
        chunks = gpd.overlay(region, int_region, how='intersection')

        # Identify leftover areas, which may result from boundary misalignment
        # or from partial coverage of intermediate regions (e.g., only urban or rural areas).
        leftover_areas = region.copy()
        chunks_union = chunks.geometry.unary_union
        leftover_areas.geometry = region.geometry.difference(chunks_union)
        leftover_areas = leftover_areas.explode(ignore_index=True)
        leftover_areas = leftover_areas[~leftover_areas.is_empty]

        combined = pd.concat([chunks, leftover_areas], axis=0)
        
        # Identify and merge thin areas (or slivers) likely caused by boundary mismatches.
        # Polygons wider than the max_width threshold are considered valid and retained.
        cleaned_split = self.merge_thin_areas(combined, max_width=0.5)
                
        if verbose:
            print("Number of chunks:", len(cleaned_split))

        return cleaned_split
    
    def _filter_polygons_in_region(self, input_polys, region, verbose=False):
        """Filter polygons to only those within the target region."""
        centroids = input_polys.copy()
        centroids['geometry'] = centroids['geometry'].apply(lambda geom: geom.representative_point())
        centroids = centroids.sjoin(region, how="inner", predicate="intersects").copy()
        input_polys = input_polys.merge(centroids[[self.poly_id]], on=self.poly_id, how="inner").copy()

        if verbose:
            print("-----------------------------------------------")
            print("Region Code:", self.region_id, " - ",
                  "Name:", region.commune.iloc[0], "\n")
        return input_polys.copy()
        
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
            centroids[[self.poly_id, subregion_id]],
            on=self.poly_id, 
            how="left").copy()
        
        return self.data.copy()
        
    def _densify_polygons(self, distance):
        """Densify polygon vertices for better Voronoi results."""
        self.data = self.densifier.densify_geodataframe(self.data, distance)
        return self.data
    
    def _create_voronoi_diagram(self, input_data, region, buffer_region=10):
        """Create constrained Voronoi diagram."""
        # Apply a buffer to the simplified geometries region geometry to mitigate gaps introduced during simplification,
        # ensuring alignment with the original region boundaries after clipping
        region_buffered = region.copy()
        region_buffered["geometry"] = region_buffered.geometry.buffer(buffer_region)
        voronoi = voronoiDiagram4plg(input_data, region_buffered)
        return voronoi
    
    def _process_hidden_polygons(self, voronoi, apply_overlay=False, verbose=False):
        """
        Handle hidden/overlapped polygons in Voronoi diagram.

        Args:
            voronoi (GeoDataFrame): The Voronoi diagram to process.
            apply_overlay (bool): If True, overlays hidden polygons on visible ones.
            return_hidden (bool): If True, also returns hidden polygons as a second output.

        Returns:
            GeoDataFrame: Processed Voronoi polygons (with or without overlay).
            GeoDataFrame (optional): Hidden polygons, only if return_hidden is True.
        """
        _, hidden_indices = self.hidden_processor.find_hidden_polygons(voronoi)
        hidden_gdf = voronoi[voronoi.index.isin(hidden_indices)]
        if verbose:
            print("No. of hidden polygons:", len(hidden_gdf))

        if not hidden_gdf.empty:
            visible = voronoi[~voronoi.index.isin(hidden_indices)]

            if apply_overlay:
                union = gpd.overlay(visible, hidden_gdf, how='union')
                # Merge attributes: prioritise hidden (_2) values, fallback to visible (_1)
                for col in visible.columns:
                    if col != 'geometry':
                        col_1, col_2 = f"{col}_1", f"{col}_2"
                        if col_1 in union.columns and col_2 in union.columns:
                            union[col_2] = union[col_2].fillna(union[col_1])
                            union[col] = union[col_2]
                            union.drop(columns=[col_1, col_2], inplace=True)
                            if col == "commune_id":
                                union[col] = union[col].astype(int)
                return union, hidden_gdf
            return visible, hidden_gdf

        return voronoi, gpd.GeoDataFrame(columns=voronoi.columns, crs=voronoi.crs)
    
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
        topo = tp.Topology(gdf, prequantize=False)
        simplified = topo.toposimplify(tolerance).to_gdf()

        clipped = gpd.clip(simplified, region_gdf)
        return clipped
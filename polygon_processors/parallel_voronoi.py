from joblib import Parallel, delayed
import geopandas as gpd
import pandas as pd

class ParallelVoronoiProcessor:
    """
    A class to parallelise the processing of Voronoi diagrams by chunks using joblib.
    
    This class is designed to be used with VoronoiProcessor to handle the parallel
    execution of the chunked processing.
    """
    
    def __init__(self, n_jobs=-1, verbose=1):
        """
        Initialise the parallel processor.
        
        Args:
            n_jobs (int): Number of parallel jobs to run. -1 means using all processors.
            verbose (int): Controls the verbosity of joblib's parallel execution.
        """
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def process_chunk(self, voronoi_processor, chunk_poly, region, buffer_region, 
                     process_hidden, overlay_hidden, simplify_bdry, tolerance,
                     verbose):
        """
        Process a single chunk of the region.
        
        This method contains the core chunk processing logic that will be parallelised.
        """
        chunk = gpd.GeoDataFrame([chunk_poly], crs=region.crs)
        chunk_data = voronoi_processor.data
        chunk_data = voronoi_processor._filter_polygons_in_region(chunk_data, chunk, False)

        if chunk_data.empty:
            if verbose:
                print("Skipping chunk (no polygons)")
            return None, None

        if verbose:
            print(f"Processing chunk region with {len(chunk_data)} polygons")

        vor = voronoi_processor._create_voronoi_diagram(chunk_data, chunk, buffer_region)

        hidden_gdf = None
        if process_hidden:
            vor, hidden_gdf = voronoi_processor._process_hidden_polygons(
                vor,
                overlay_hidden, 
                False
        )

        if simplify_bdry:
            try:
                vor = voronoi_processor._simplify_boundaries(vor, chunk, tolerance)
            except Exception as e:
                print("❌ Error in chunk:", e)
                raise e

        return vor, hidden_gdf
    
    def parallel_process(self, voronoi_processor, region_chunks, region, buffer_region,
                        process_hidden, overlay_hidden, simplify_bdry, tolerance,
                        verbose):
        """
        Parallel processing of all region chunks.
        
        Args:
            voronoi_processor (VoronoiProcessor): The main processor instance.
            region_chunks (GeoDataFrame): The chunks to process in parallel.
            region (GeoDataFrame): The full region geometry.
            buffer_region (float): Buffer distance for region clipping.
            process_hidden (bool): Whether to process hidden polygons.
            overlay_hidden (bool): Whether to overlay hidden polygons.
            simplify_bdry (bool): Whether to simplify boundaries.
            tolerance (float): Simplification tolerance.
            verbose (bool): Whether to show progress messages.
            
        Returns:
            tuple: (combined Voronoi results, combined hidden polygons)
        """
        results = Parallel(n_jobs=self.n_jobs, verbose=False)(
            delayed(self.process_chunk)(
                voronoi_processor,
                chunk_poly,
                region,
                buffer_region,
                process_hidden,
                overlay_hidden,
                simplify_bdry,
                tolerance,
                verbose
            )
            for _, chunk_poly in region_chunks.iterrows()
        )
        
        # Unpack results
        voronoi_chunks = [
            res[0] for res in results
            if res[0] is not None and not res[0].empty
        ]

        hidden_chunks = [
            res[1] for res in results
            if (
                res[1] is not None
                and not res[1].empty
                and not res[1].isna().all().all()
            )
        ]
        
        # Combine results
        if voronoi_chunks:
            voronoi = gpd.GeoDataFrame(
                pd.concat(voronoi_chunks, ignore_index=True),
                crs=region.crs
            )
        else:
            voronoi = gpd.GeoDataFrame(
                columns=voronoi_processor.data.columns,
                crs=region.crs
            )
        
        hidden_gdf = None
        if hidden_chunks:
            hidden_gdf = gpd.GeoDataFrame(pd.concat(hidden_chunks, ignore_index=True), crs=region.crs)
        
        return voronoi, hidden_gdf
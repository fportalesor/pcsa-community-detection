import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained
from longsgis import voronoiDiagram4plg
from .base_processor import PolygonProcessor
from .hidden_polys import HiddenPolygonProcessor

class PolygonSplitter(PolygonProcessor):
    """
    Splits polygons with high population into smaller clusters using the geocoded points.
    Inherits from PolygonProcessor for common functionality.
    """
    
    def __init__(self, input_data=None, id_points="patient_id",pop_max=300, split_polygons=False, 
                 tolerance=0.2, root_folder=None):
        """
        Initialise the split processor.
        
        Args:
            input_data (GeoDataFrame): Input polygons to process
            POP_max (int): Maximum population per cluster
            root_folder (str/Path): Root directory path
        """
        super().__init__(root_folder)
        self.data = input_data
        self.pop_max = pop_max
        self.id_points = id_points
        self.split_polygons=split_polygons
        self.tolerance = tolerance
        self.hidden_processor = HiddenPolygonProcessor()
    
    def _split_single_building_block(self, points, building_block_id):
        """Split a single polygon into clusters based on point distribution."""
        # Get the polygon to split
        poly = self.data[self.data["MANZENT"] == building_block_id].copy()
    
        # Perform spatial join to get points within the polygon
        points = points.sjoin(poly, how="inner", predicate="intersects")
    
        # Check if we have enough points to cluster
        if len(points) == 0:
            return None  # or handle empty case as needed
    
        # Prepare for clustering - use point coordinates directly
        X = np.column_stack((points.geometry.x, points.geometry.y))
    
        # Determine cluster count based on population
        population = poly["POP"].iloc[0]

        n_clusters = (round(population / self.pop_max))
        n_clusters = 2 if n_clusters < 2 else n_clusters

        pop_min = round((population/n_clusters) * (1 - self.tolerance))
        pop_max = round((population/n_clusters) * (1 + self.tolerance))

        #print("total_pop:", population)
        #print("n_clusters:", n_clusters)
        #print("pop_min:", pop_min)
        #print("pop_max:", pop_max)
    
        # Perform clustering if we have enough points
        if len(points) >= n_clusters:

            kmeans = KMeansConstrained(
                n_clusters=n_clusters,
                size_min=pop_min,       
                size_max=pop_max,  
                random_state=42
            )
            points["cluster"] = kmeans.fit_predict(X) + 1
        else:
            # If not enough points for clustering, assign all to one cluster
            points["cluster"] = 0
    
        # Clean up columns
        for col in ["index_left", "index_right"]:
            if col in points.columns:
                points = points.drop(columns=[col])

        # Create Voronoi polygons
        vd = voronoiDiagram4plg(points, poly)
        vd = vd[["geometry", "cluster", self.id_points]]

        # Calculate POP for each Voronoi polygon
        pop_counts = (
            vd
            .groupby("cluster")[self.id_points].nunique()
            .reset_index(name="POP")
        )

        # Dissolve voronoi polygons by cluster
        dissolved = vd.dissolve(by="cluster").reset_index()

        # Merge counts
        dissolved = dissolved.merge(pop_counts, on="cluster", how="left")        

        # Merge with original attributes
        dissolved["MANZENT"] = building_block_id
        for col in ["CUT", "COMUNA", "TIPO_ZONA"]:
            dissolved[col] = poly[col].iloc[0]
            
        # Format output
        dissolved["cluster"] = dissolved["cluster"].astype(str).str.zfill(2)
        dissolved["MANZENT"] = dissolved["MANZENT"] + dissolved["cluster"]

        return dissolved[['CUT', 'COMUNA', 'MANZENT', 'POP','TIPO_ZONA', 'geometry']]
    
    def _split_building_blocks(self, points):

        # Filter polygons needing splitting
        to_split = self.data[self.data["POP"] > self.pop_max].copy()
        if len(to_split) == 0:
            return self.data
            
        # Process each polygon
        results = []
        for building_block_id in to_split["MANZENT"].unique():
            print(f"Processing Building Block: {building_block_id}")
            results.append(self._split_single_building_block(points, building_block_id))
        
        # Combine results
        split_polys = pd.concat(results, ignore_index=True)

        # Identify overlapping or hidden polygons in the splitted voronoi polys
        _, hidden_indices = self.hidden_processor.find_hidden_polygons(split_polys)
        hidden_gdf = split_polys[split_polys.index.isin(hidden_indices)]
        print("N° of hidden polygons:",len(hidden_gdf))
        
        if not hidden_gdf.empty:
            hidden_gdf.to_file(self.root_folder / f"processed_data/splitted_hidden_polys.shp")
        
        remaining_polys = self.data[~self.data["MANZENT"].isin(to_split["MANZENT"])]
        
        self.data = pd.concat([remaining_polys, split_polys], ignore_index=True)
        self.data["MANZENT"] = self.data["MANZENT"].str.ljust(18, fillchar='0')

        return self.data
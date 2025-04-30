import pandas as pd
from .base_processor import PolygonProcessor

class MultipartPolygonProcessor(PolygonProcessor):
    """
    Processor for handling multipart polygons, converting them to single-part.
    Inherits from PolygonProcessor for common functionality.
    """
    
    def __init__(self, input_data=None, root_folder=None):
        """
        Initialise the multipart processor with input data.
        
        Args:
            input_data (GeoDataFrame, optional): Input polygon data. 
                Defaults to None (can be loaded later).
            root_folder (str/Path, optional): Root directory path. 
                Defaults to None (uses parent of current file).
        """
        super().__init__(root_folder)
        self.data = input_data
        
    def process(self):
        """
        Process multipart polygons by exploding them
        
        Args:
                
        Returns:
            GeoDataFrame: Processed polygons
        """
        if self.data is None:
            raise ValueError("No input data provided. Load data first.")
            
        # Convert multipart to singlepart polygons
        self._explode_multipart_polygons()
        
        # Process duplicated polygons (originally multipart)
        polys = self._identify_duplicated_polygons()
        
        return polys
    
    def _explode_multipart_polygons(self):
        """Convert multipart polygons to singlepart and clean up indexes."""
        self.data = self.data.explode(index_parts=True)
        self.data = self.data.reset_index(drop=True)
        
    def _identify_duplicated_polygons(self):
        """
        Identify blocks/entities with multiple polygons and create new IDs.
        
        Returns:
            GeoDataFrame: Duplicated polygons with new IDs
        """
        # Find duplicated polygons (keeping all occurrences)
        dup = self.data.loc[self.data.duplicated(subset=['MANZENT'], keep=False)].copy()

        # Create sequential counts for each duplicated ID
        dup['count'] = dup.groupby("MANZENT").cumcount() + 1
        dup = dup.reset_index(drop=True)

        # Create new unique IDs by appending count
        dup['count'] = dup['count'].astype(str).str.zfill(2)
        dup["MANZENT"] = dup["MANZENT"] + dup["count"]

        # Combine with non-duplicated polygons
        original_ids = dup["MANZENT"].str[:-2]
        self.data = self.data.loc[~self.data["MANZENT"].isin(original_ids)]
        self.data = pd.concat([self.data, dup], axis=0)
        self.data = self.data.reset_index(drop=True)

        # Ensure ID length is 16 characters, padding with zeros if needed
        self.data["MANZENT"] = self.data["MANZENT"].str.ljust(16, fillchar='0')

        return self.data[["CUT", "COMUNA", "MANZENT", "TIPO_ZONA","geometry"]]
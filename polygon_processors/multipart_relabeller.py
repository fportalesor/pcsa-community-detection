import pandas as pd
from .base_processor import PolygonProcessor

class MultipartPolygonRelabeller(PolygonProcessor):
    """
    Specialised processor for handling multipart polygons.

    Processor to identify multipart polygons and assign new unique IDs by appending counts.

    Multipart polygons in the 2017 Chilean Census data (INE) represent areas with 
    1 to 3 private homes combined into single polygons.

    Args:
        data (Any): Placeholder for loading or storing data (e.g., a GeoDataFrame).
        poly_id (str): Name of the column used to uniquely identify polygons.
        id_length (int): Fixed length for polygon IDs, padded with zeros if needed.
    """
    
    def __init__(self, input_data=None, id_length=16, poly_id="block_id"):
        super().__init__(poly_id)
        self.data = input_data
        self.id_length = id_length
        
    def _relabel_multipart_blocks(self):
        """
        Identify blocks/entities with multiple polygons and create new IDs.
        
        Returns:
            GeoDataFrame: Duplicated polygons with new IDs
        """
        
        self.data, dup = self.identify_multipart_polygons(
            self.data, self.poly_id)

        # Create sequential counts for each duplicated ID
        dup['count'] = dup.groupby(self.poly_id).cumcount() + 1
        dup = dup.reset_index(drop=True)

        # Create new unique IDs by appending count
        dup['count'] = dup['count'].astype(str).str.zfill(2)
        dup[self.poly_id] = dup[self.poly_id] + dup["count"]

        # Combine with non-duplicated polygons
        original_ids = dup[self.poly_id].str[:-2]
        self.data = self.data.loc[~self.data[self.poly_id].isin(original_ids)]
        self.data = pd.concat([self.data, dup], axis=0)
        self.data = self.data.reset_index(drop=True)

        # Ensure ID length is 16 characters, padding with zeros if needed
        self.data[self.poly_id] = self.data[self.poly_id].str.ljust(self.id_length, fillchar='0')

        expected_cols = ["commune_id", "commune", self.poly_id, "zone_type", "geometry"]
        available_cols = [col for col in expected_cols if col in self.data.columns]

        return self.data[available_cols].copy()
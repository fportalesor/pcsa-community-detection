import pandas as pd
from .base_processor import PolygonProcessor

class MultipartPolygonProcessor(PolygonProcessor):
    """
    Specialised processor for handling multipart polygons.
    """
    
    def __init__(self, input_data=None, root_folder=None, id_column=None):
        """
        Initialise the multipart processor with input data.
        
        Attributes:
            root_folder (Path): Path to the root directory for data storage or outputs.
            data (Any): Placeholder for loading or storing data (e.g., a GeoDataFrame).
            id_column (str): Name of the column used to uniquely identify polygons.
        """
        super().__init__(root_folder, id_column)
        self.data = input_data
        
    def _relabel_multipart_blocks(self):
        """
        Identify blocks/entities with multiple polygons and create new IDs.
        
        Returns:
            GeoDataFrame: Duplicated polygons with new IDs
        """
        
        self.data, dup = self.identify_multipart_polygons(
            self.data, self.id_column)

        # Create sequential counts for each duplicated ID
        dup['count'] = dup.groupby(self.id_column).cumcount() + 1
        dup = dup.reset_index(drop=True)

        # Create new unique IDs by appending count
        dup['count'] = dup['count'].astype(str).str.zfill(2)
        dup[self.id_column] = dup[self.id_column] + dup["count"]

        # Combine with non-duplicated polygons
        original_ids = dup[self.id_column].str[:-2]
        self.data = self.data.loc[~self.data[self.id_column].isin(original_ids)]
        self.data = pd.concat([self.data, dup], axis=0)
        self.data = self.data.reset_index(drop=True)

        # Ensure ID length is 16 characters, padding with zeros if needed
        self.data[self.id_column] = self.data[self.id_column].str.ljust(16, fillchar='0')

        return self.data[["commune_id", "commune", self.id_column, "zone_type", "geometry"]]
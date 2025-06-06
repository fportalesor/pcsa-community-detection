import pandas as pd
import geopandas as gpd
from shapely.validation import make_valid
from .base_processor import PolygonProcessor

class PolygonDissolver(PolygonProcessor):
    """
    Dissolve polygons based on assigned TractID or Output Area definitions.

    Args:
        input_data (GeoDataFrame, optional): Preloaded GeoDataFrame with Voronoi polygons.
        root_folder (Path, optional): Root folder for file references.
    """

    def __init__(self, input_data=None, 
                 poly_id= "block_id", tract_id="TractID"):
        self.data = input_data
        self.poly_id = poly_id
        self.tract_id = tract_id

    def process(
        self,
        voronoi_path=None,
        tracts_path="../AZTool/TractOutput_150.csv",
        aztool_ids_path="../AZTool/voronoi.pat"
    ):
        """
        Processes input polygon data and dissolves them into Tracts using AZTool output.

        Args:
            voronoi_path (Path, optional): Path to the Voronoi shapefile or ZIP.
            tracts_path (Path): Path to the tract assignments CSV.
            aztool_ids_path (Path): Path to AZTool block-to-ID mapping .pat file.

        Returns:
            GeoDataFrame: Dissolved polygons with population and metadata.
        """
        if self.data is None:
            if voronoi_path is None:
                raise ValueError("No Voronoi data provided.")
            if not voronoi_path.exists():
                raise FileNotFoundError(f"Voronoi data not found at: {voronoi_path}")
            self.data = gpd.read_file(voronoi_path)

        # Load mappings
        tract_assignments = self._load_tract_assignments(tracts_path)
        aztool_ids = self._load_aztool_ids(aztool_ids_path)

        # Extract target population from filename
        # It is assumed that the filename contains a digit 
        # representing the target population per zone.
        try:
            target_pop = int(''.join(filter(str.isdigit, tracts_path.stem)))
            print(f"\nTarget population per zone: {target_pop}")
        except ValueError:
            target_pop = None
            print("\nCould not extract target population from filename.")

        # Merge assignments and dissolve polygons
        tracts = tract_assignments.merge(aztool_ids, on="AZM_ID", how="left")
        tracts = tracts[[self.poly_id, self.tract_id]]
        tracts = self.data.merge(tracts, on=self.poly_id, how="left")

        tracts_dissolved = tracts.dissolve(
            by=self.tract_id,
            aggfunc={
                "commune_id": "first",
                "commune": "first",
                "pop": "sum",
                "pop_high": "sum",
                "pop_middle": "sum",
                "pop_low": "sum"
            }
        ).reset_index()

        tracts_dissolved = tracts_dissolved[
            ["commune_id", "commune", self.tract_id,
             "pop", "pop_high", "pop_middle", "pop_low", "geometry"]
        ]

        # Print population summary
        print("\nStatistical summary for target population:")
        print(tracts_dissolved["pop"].describe().round(2))

        # Apply gap filling
        print("\nFixing internal gaps...")
        tracts_dissolved.geometry = (
            tracts_dissolved.geometry
            .apply(lambda geom: self.fill_holes(geom, sizelim=10)))

        tracts_dissolved["geometry"] = tracts_dissolved.geometry.apply(make_valid)

        return tracts_dissolved

    def _load_tract_assignments(self, path=None):
        """
        Load tract assignments and rename AZTool ID column.

        Args:
            path (Path): Path to the CSV with tract assignments.

        Returns:
            DataFrame: Cleaned assignments with standardised column names.
        """
        tracts = pd.read_csv(path)
        return tracts.rename(columns={"BldBlID": "AZM_ID"})

    def _load_aztool_ids(self, path="None"):
        """
        Load block ID mappings from AZTool output.

        Args:
            path (Path): Path to the CSV with AZTool block ID data.

        Returns:
            DataFrame: Mapped AZTool IDs and block IDs.
        """
        ids = pd.read_csv(path, dtype={self.poly_id: str})
        return ids[["AZM_ID", self.poly_id]]
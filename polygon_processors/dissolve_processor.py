import pandas as pd
import geopandas as gpd
from .base_processor import PolygonProcessor

class PolygonDissolver(PolygonProcessor):
    """
    Dissolve polygons based on assigned TractID.
    """

    def __init__(self, input_data=None, poly_id= "block_id"):
        self.data = input_data
        self.poly_id = poly_id

    def process_aztool_outcomes(
        self,
        voronoi_path=None,
        tracts_path="../AZTool/TractOutput_150.csv",
        tract_id="TractID",
        aztool_ids_path="../AZTool/voronoi.pat",
        target_col="pop",
        fill_holes=True,
        remove_dangles=True,
        calculate_stats=True,
        verbose=True
    ):
        """
        Processes input polygon data and dissolves them into Tracts using AZTool output.

        Args:
            voronoi_path (Path, optional): Path to the Voronoi shapefile or ZIP.
            tracts_path (Path): Path to the tract assignments CSV.
            tract_id (str): Column name for the dissolve grouping identifier (default: 'TractID').
            aztool_ids_path (Path): Path to AZTool block-to-ID mapping .pat file.
            target_col (str): Name of the attribute column to aggregate (default: 'pop').
            fill_holes (bool): Whether to fill small internal holes in the polygons.
            remove_dangles (bool): Whether to remove inward-facing dangles in polygon boundaries.
            calculate_stats (bool): Whether to calculate descriptive statistics for a specified column.
            verbose (bool): If True, prints intermediate information during processing.

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
            if verbose:
                print(f"\nTarget population per zone: {target_pop}")
        except ValueError:
            target_pop = None
            if verbose:
                print("\nCould not extract target population from filename.")

        # Merge assignments and dissolve polygons
        tracts = tract_assignments.merge(aztool_ids, on="AZM_ID", how="left")
        tracts = tracts[[self.poly_id, tract_id]]

        tracts = self.data.merge(tracts, on=self.poly_id, how="left")
        
        tracts_dissolved = tracts.dissolve(
            by=tract_id,
            aggfunc={
                "commune_id": "first",
                "commune": "first",
                target_col: "sum",
                "sregion_id": "first",
                "pop_high": "sum",
                "pop_middle": "sum",
                "pop_low": "sum"
            }
        ).reset_index()

        tracts_dissolved = tracts_dissolved[
            ["commune_id", "commune", tract_id, "sregion_id",
             target_col, "pop_high", "pop_middle", "pop_low", "geometry"]
        ]

        if verbose:
            print("\nStatistical summary for target population:")
            print(tracts_dissolved[target_col].describe().round(2))
            
        if fill_holes:
            if verbose:
                print("\nFixing internal holes...")
            tracts_dissolved.geometry = (
                tracts_dissolved.geometry
                .apply(lambda geom: self.fill_holes(geom, sizelim=50))
            )
        
        if remove_dangles:
            if verbose:
                print("\nRemoving dangles...")
            tracts_dissolved.geometry = tracts_dissolved.geometry.apply(
                self.remove_dangles
            )

        if calculate_stats:
            stats = self._calculate_stats(tracts_dissolved, target_col, target_pop)

            return tracts_dissolved, stats
        
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

    def _calculate_stats(self, gdf, variable, value):
        """
        Calculate descriptive statistics for a specified column in a GeoDataFrame.

        Args:
            gdf (GeoDataFrame): Input GeoDataFrame.
            variable (str): Column name for which to compute statistics.

        Returns:
            DataFrame: A single-row DataFrame with summary statistics.
        """
        if variable in gdf.columns:
            stats = gdf[variable].describe().to_frame().T
            stats[f"target_{variable}"] = value
        else:
            print(f" '{variable}' column not found in layer")

        stats = stats[[f"target_{variable}"] + 
                      [col for col in stats.columns if col != f"target_{variable}"]]
        
        return stats
    
    def _concat_stats(self, stats_list, variable):
        """
        Concatenate a list of statistics DataFrames and format the result.

        Args:
            stats_list (list): List of DataFrames returned from _calculate_stats.

        Returns:
            DataFrame: Concatenated and formatted DataFrame with multi-level columns.
        """
        # Filter out any empty DataFrames
        stats_list = [df for df in stats_list if not df.empty]
        if not stats_list:
            return pd.DataFrame()

        combined = pd.concat(stats_list, ignore_index=True)

        if "count" in combined.columns:
            combined = combined.rename(columns={"count": "polys_count"})

        target_col = f"target_{variable}"
        cols = [target_col] + [c for c in combined.columns if c != target_col]
        combined = combined[cols]
    
        new_columns = []
        for col in combined.columns:
            if col in [target_col, "polys_count"]:
                new_columns.append(col)
            else:
                new_columns.append(f"{variable}_{col}")
        combined.columns = new_columns
        
        combined["polys_abs_change"] = combined["polys_count"].diff().astype("Int64")
        combined["polys_pct_change"] = combined["polys_count"].pct_change().round(3) * 100

        return combined
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from .split_processor import PolygonSplitter

class AttributeCalculator(PolygonSplitter):
    """
    Calculates population and socioeconomic attributes for polygons.
    """

    def __init__(self, input_data=None, points_id="id", poly_id="block_id", 
                 pop_max=150, split_polygons=False, tolerance=0.2, 
                 cluster_col="cluster", pop_col="pop",
                 resolution=12, buffer_distance=400, target_crs=32719, n_jobs=8, verbose=True):
        """
        Initialise the AttributeCalculator.

        Args:
            input_data (GeoDataFrame): Polygon input data with 'pop' attribute.
            points_id (str): Column name identifying input points.
            poly_id (str): Column name identifying polygons.
            pop_max (int): Population threshold for splitting.
            split_polygons (bool): Whether to split polygons exceeding threshold.
            tolerance (float): Allowed variation in population size across clusters.
            cluster_col (str): Column name to assign cluster IDs.
            pop_col (str): Column name for storing population counts. Defaults to 'pop'.
            resolution (int): H3 resolution used for grid generation.
            buffer_distance (float): Distance to buffer polygons when generating H3 grids.
            target_crs (int): Target EPSG code for spatial operations.
            n_jobs (int): Number of parallel jobs for the split process.
            verbose (bool): Flag to enable progress messages.
        """
        self.data = input_data
        self.pop_max = pop_max
        self.points_id = points_id
        self.poly_id = poly_id
        self.cluster_col = cluster_col
        self.pop_col = pop_col
        self.split_polygons = split_polygons
        self.tolerance = tolerance
        self.resolution = resolution
        self.buffer_distance = buffer_distance
        self.target_crs = target_crs
        self.n_jobs = n_jobs
        self.verbose = verbose

    def process(self, 
                geocoded_data=None,
                urban_se_data_path="data/ISMT_2017_Zonas_Censales.zip",
                rural_se_data_path="data/ISMT_2017_Localidades_Rurales.zip"):
        """
        Main processing function to calculate population and socioeconomic attributes.

        Args:
            geocoded_data (str, Path, or GeoDataFrame): Geocoded population data.
            urban_se_data_path (str): File path to urban socioeconomic data.
            rural_se_data_path (str): File path to rural socioeconomic data.

        Returns:
            GeoDataFrame: Polygon dataset with population and socioeconomic attributes.
        """
        points = self._load_geocoded_data(geocoded_data)
        self.data = self._calculate_population(points)
        points_in_polygons = self.data[self.pop_col].sum()

        if self.verbose:
            print(f"Points within study area (intersecting polygons): {points_in_polygons:,}")

        if self.split_polygons:
            self.data = self._split_building_blocks(points)
            self.data = self._calculate_population(points)

            points_in_polygons = self.data[self.pop_col].sum()
            if self.verbose:
                print(f"Points after polygon splitting: {points_in_polygons:,}")

        se_data = self._load_socioeconomic_data(urban_se_data_path, rural_se_data_path)
        self.data = self._add_socioeconomic_groups(se_data)

        if self.verbose:
            print("\nPopulation statistics (only points within study area):")
            print(self.data[self.pop_col].describe().round(2))
        return self.data

    def _load_geocoded_data(self, geocoded_data):
        """
        Load geocoded data from either a file path or existing GeoDataFrame.

        Args:
            geocoded_data (str, Path, or GeoDataFrame): Input data.

        Returns:
            GeoDataFrame: Validated point data.
        """
        if isinstance(geocoded_data, (str, Path)):
            points = gpd.read_file(geocoded_data)
        elif isinstance(geocoded_data, gpd.GeoDataFrame):
            points = geocoded_data.copy()
        else:
            raise ValueError("Input must be either a file path or GeoDataFrame")

        points = self._validate_crs(points)
        return points

    def _load_socioeconomic_data(self, urban_se_data_path, rural_se_data_path):
        """
        Load and process socioeconomic data for urban and rural zones.

        This function loads geospatial data containing estimated counts of individuals 
        in three socioeconomic groups —low, middle, and high— based on the ISMT Index 
        (Socio-Material Territorial Indicator) developed by the Observatorio de 
        Ciudades UC (OCUC), Chile. The original variables are renamed for consistency,
        and proportions for each group are calculated relative to the total estimated
        population in each zone.

        Args:
            urban_se_data_path (str or Path): File path to the socioeconomic data for urban areas.
            rural_se_data_path (str or Path): File path to the socioeconomic data for rural areas.

        Returns:
            GeoDataFrame: Combined socioeconomic dataset with columns:
                - 'zone': Zone identifier
                - 'pct_low': Proportion of population in the low socioeconomic group
                - 'pct_middle': Proportion of population in the middle socioeconomic group
                - 'pct_high': Proportion of population in the high socioeconomic group

        Note:
            For more information, see: https://ismtchile.geocoded.dev/home or 
            https://github.com/cran/ismtchile
        """
        urban = gpd.read_file(urban_se_data_path)
        rural = gpd.read_file(rural_se_data_path)

        rename_dict = {'zona': 'zone', 'Alto': 'high', 'Medio': 'middle', 'Bajo': 'low'}
        urban = urban.rename(columns=rename_dict)[['zone', 'high', 'middle', 'low']]
        rural = rural.rename(columns=rename_dict)[['zone', 'high', 'middle', 'low']]

        se_data = pd.concat([urban, rural])
        se_data['total'] = se_data[['high', 'middle', 'low']].sum(axis=1)

        for group in ['high', 'middle', 'low']:
            se_data[f'pct_{group}'] = (se_data[group] / se_data['total']).fillna(0)

        return se_data.drop(columns=['total', 'high', 'middle', 'low'])

    def _calculate_population(self, geocoded_data):
        """
        Assigns population points to polygons, resolving duplicates.

        Args:
            self.data: Voronoi polygons
            geocoded_data: Point data representing population counts

        Returns:
            GeoDataFrame with population counts per polygon
        """
        intersection = gpd.sjoin(geocoded_data, self.data[[self.poly_id, 'geometry']], how='inner', predicate='intersects')
        
        duplicate_points = intersection[self.points_id].duplicated()

        if duplicate_points.any():
            duplicates = intersection[duplicate_points]
            nearest = gpd.sjoin_nearest(
                geocoded_data[geocoded_data[self.points_id].isin(duplicates[self.points_id].unique())],
                self.data[[self.poly_id, 'geometry']], how='inner')
            non_duplicates = intersection[~duplicate_points]
            final_assignment = pd.concat([
                non_duplicates[[self.poly_id, self.points_id]],
                nearest[[self.poly_id, self.points_id]]
            ])
        else:
            final_assignment = intersection[[self.poly_id, self.points_id]]

        pop_by_polygon = (
            final_assignment
            .groupby(self.poly_id)[self.points_id]
            .nunique()
            .reset_index(name=self.pop_col)
        )

        self.data = self.data.merge(pop_by_polygon, on=self.poly_id, how='left')
        self.data[self.pop_col] = self.data[self.pop_col].fillna(0).astype(int)
        return self.data

    def round_preserve_sum(self, row, groups=None):
        """
        Round population groups while preserving total, handling edge cases.
        """
        try:
            values = [row[f'{self.pop_col}_{group}'] for group in groups]

            # Handle NaN/Inf values
            if any(not np.isfinite(v) for v in values):
                return pd.Series([0] * len(groups), index=groups)

            values = np.array(values, dtype=np.float64)
            total = float(row[self.pop_col])

            rounded = np.round(values).astype(np.int64)
            diff = int(total - rounded.sum())

            if diff != 0:
                largest_idx = np.argmax(values)
                rounded[largest_idx] += diff

                if rounded[largest_idx] < 0:  # Avoid negative population
                    rounded = np.floor(values).astype(np.int64)
                    rounded[largest_idx] += int(total - rounded.sum())

            return pd.Series(rounded, index=groups)

        except Exception as e:
            if self.verbose:
                print(f"Error processing row: {row}\nError: {str(e)}")
            return pd.Series([0] * len(groups), index=groups)

    def _add_socioeconomic_groups(self, se_data):
        """
        Estimates population counts per socioeconomic group in each polygon.

        Args:
            se_data (DataFrame): Socioeconomic data with group proportions.

        Returns:
            GeoDataFrame: Polygons with estimated group populations.
        """
        self.data["zone"] = self.data[self.poly_id].str[:11]
        self.data = self.data.merge(se_data, on="zone", how="left").drop(columns="zone")

        groups = ['pct_high', 'pct_middle', 'pct_low']
        self.data[groups] = self.data[groups].fillna(0)

        all_zero_mask = (self.data[groups] == 0).all(axis=1)
        equal_value = 1 / len(groups)
        self.data.loc[all_zero_mask, groups] = equal_value

        for group in groups:
            self.data[f'{self.pop_col}_{group}'] = self.data[self.pop_col] * self.data[group]

        rounded = self.data.apply(lambda row: self.round_preserve_sum(row, groups=groups), axis=1)
        for group in groups:
            self.data[f'{self.pop_col}_{group}'] = rounded[group]

        self.data = self.data.drop(columns=groups)

        rename_dict = {f'{self.pop_col}_{group}': f'{self.pop_col}_{group.split("_")[1]}' for group in groups}
        self.data = self.data.rename(columns=rename_dict)

        return self.data
import geopandas as gpd
import pandas as pd
import numpy as np
from .split_processor import PolygonSplitter

class AttributeCalculator(PolygonSplitter):


    def __init__(self, input_data=None, id_points="patient_id",pop_max=300, split_polygons=False, 
                 tolerance=0.2, root_folder=None):
        """
        """
        super().__init__(root_folder)
        self.data = input_data
        self.pop_max = pop_max
        self.id_points = id_points
        self.split_polygons=split_polygons
        self.tolerance = tolerance
    
    def process(self, 
                geocoded_data_path="processed_data/syn_pop.shp",
                se_data_path_urban="data/ISMT_2017_Zonas_Censales.zip",
                se_data_path_rural="data/ISMT_2017_Localidades_Rurales.zip"):
        """
        """
        
        # Load geocoded data
        points = self._load_geocoded_data(geocoded_data_path)

        # Calculate Population
        self.data = self._calculate_population(points)
        # Total pop
        print("total pop:", self.data.POP.sum())

        # Split
        if self.split_polygons:
            self.data = self._split_building_blocks(points)

        # Add socioeconomic groups
        se_data = self._load_socioeconomic_data(se_data_path_urban,
                                                se_data_path_rural)
        
        self.data = self._add_socioeconomic_groups(se_data)

        print(self.data['POP'].describe())

        return self.data
    

    def _load_geocoded_data(self, geocoded_data_path):
        """Load geocoded data."""
        points = gpd.read_file(geocoded_data_path)
        points = self._validate_crs(points)


        return points
    
    def _load_socioeconomic_data(self, se_data_path_urban, se_data_path_rural):
        """Load Socioeconomic data."""
        cols_list = ['zona', 'Alto', 'Medio', 'Bajo']
        se_data_urban = gpd.read_file(se_data_path_urban)
        se_data_rural = gpd.read_file(se_data_path_rural)

        se_data_urban = se_data_urban[cols_list]
        se_data_rural = se_data_rural[cols_list]

        se_data = pd.concat([se_data_urban, se_data_rural], axis=0)

        # Calculate total count per row (sum of all 3 groups)
        se_data['total'] = se_data[['Alto', 'Medio', 'Bajo']].sum(axis=1)

        # Calculate percentage for each group (handling division by zero)
        for group in ['Alto', 'Medio', 'Bajo']:
            se_data[f'pge_{group}'] = (se_data[group] / se_data['total']).fillna(0)

        se_data = se_data.drop(columns=['total', 'Alto', 'Medio', 'Bajo'])

        return se_data
    
    def _calculate_population(self, geocoded_data):
        """
        Assign population points to Voronoi polygons, handling boundary cases by
        reassigning duplicates to their closest polygon using nearest join.
    
        Args:
            self.data: Voronoi polygons with 'MANZENT' identifiers
            geocoded_data: Point data representing population counts
    
        Returns:
            GeoDataFrame with population counts per polygon
        """
        
        # 1. Initial spatial join (keep all matches to detect duplicates)
        intersection = gpd.sjoin(
            geocoded_data, 
            self.data[['MANZENT', 'geometry']],
            how='inner', 
            predicate='intersects'
        )
    
        # 2. Identify duplicate points (assume geocoded_data has an index)
        duplicate_points = intersection[self.id_points].duplicated()
    
        if duplicate_points.any():
            # 3. Handle duplicates with nearest join
            duplicates = intersection[duplicate_points]
            unique_duplicate_points = geocoded_data.loc[duplicates[self.id_points].unique()]
        
            # Perform nearest join for duplicates
            nearest_join = gpd.sjoin_nearest(
                unique_duplicate_points,
                self.data[['MANZENT', 'geometry']],
                how='inner'
            )
        
            # 4. Create final assignment by combining:
            # - Non-duplicate points from original intersection
            # - Reassigned duplicates from nearest join
            non_duplicates = intersection[~duplicate_points]
            final_assignment = pd.concat([
                non_duplicates[['MANZENT', self.id_points]],
                nearest_join[['MANZENT', self.id_points]]
            ])
        else:
            # No duplicates found - use original intersection
            final_assignment = intersection[['MANZENT', self.id_points]]
    
        # 5. Count population per polygon
        pop_by_polygon = (
            final_assignment
            .groupby('MANZENT')[self.id_points].nunique()
            .reset_index(name="POP")
        )

        # Merge with original polygons
        self.data = self.data.merge(
            pop_by_polygon,
            on='MANZENT',
            how='left'
        )
        
        # Fill NA with 0 and convert to integer
        self.data['POP'] = self.data['POP'].fillna(0).astype(int)
    
        return self.data
    
    def _add_socioeconomic_groups(self, se_data):
        """
        Assigns socioeconomic groups to Voronoi polygons, with rounded population counts
        that preserve the original total population.
    
        Args:
            self.data (GeoDataFrame): Voronoi polygons with 'MANZENT' column
            se_data (DataFrame): Socioeconomic data with percentages
    
        Returns:
            GeoDataFrame: Updated polygons with population counts per group
        """

        # Merge data
        self.data["zona"] = self.data["MANZENT"].str[:11]
        self.data = self.data.merge(se_data, on="zona", how="left")

        self.data = self.data.drop(columns="zona")

        groups = ['pge_Alto', 'pge_Medio', 'pge_Bajo']

        # Fill NA and handle all-zeros case
        self.data[groups] = self.data[groups].fillna(0)

        # Create mask for rows where all groups are 0
        all_zero_mask = (self.data[groups] == 0).all(axis=1)
    
        # Calculate equal distribution value (1/3 for 3 groups)
        equal_value = 1 / len(groups)
    
        # Apply equal distribution where needed
        self.data.loc[all_zero_mask, groups] = equal_value

        # Estimate population counts for each group (handling division by zero)
        # Step 1: Calculate unrounded estimates
        for group in groups:
            self.data[f'POP_{group}'] = self.data['POP'] * self.data[group]

        # Step 2: Apply rounding with sum preservation
        def round_preserve_sum(row):
            """Round population groups while preserving total, handling edge cases."""
            try:
                values = [row[f'POP_{group}'] for group in groups]
        
                # Handle NaN/Inf values
                if any(not np.isfinite(v) for v in values):
                    return pd.Series([0]*len(groups), index=groups)
            
                # Convert to float64 to prevent overflow
                values = np.array(values, dtype=np.float64)
                total = float(row['POP'])
        
                # Initial rounding
                rounded = np.round(values).astype(np.int64)
                diff = int(total - rounded.sum())
        
                # Adjust largest group if needed
                if diff != 0:
                    largest_idx = np.argmax(values)
                    rounded[largest_idx] += diff
            
                    # Final overflow check
                    if rounded[largest_idx] < 0:  # Can't have negative population
                        rounded = np.floor(values).astype(np.int64)
                        rounded[largest_idx] += int(total - rounded.sum())
                
                return pd.Series(rounded, index=groups)
        
            except Exception as e:
                print(f"Error processing row: {row}\nError: {str(e)}")
                return pd.Series([0]*len(groups), index=groups)
    
        # Apply rounding correction
        rounded = self.data.apply(round_preserve_sum, axis=1)
        for group in groups:
            self.data[f'POP_{group}'] = rounded[group]

        # Delete percentage columns
        self.data = self.data.drop(columns=groups)

        # Rename population columns (remove '_pge')
        rename_dict = {
        f'POP_{group}': f'POP_{group.split("_")[1]}'
        for group in groups
        }
        self.data = self.data.rename(columns=rename_dict)
    
        return self.data 
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import LineString
from sklearn.preprocessing import MinMaxScaler

class MatrixConstructor:
    def __init__(self, tracts_gdf=None, tracts_path=None, patient_data_path=None, 
                 locations_data_path=None, health_centres_path=None, 
                 TractID="TractID", patient_id="id"):
        self.tracts_path = tracts_path
        self.patient_data_path = patient_data_path
        self.locations_data_path = locations_data_path
        self.health_centres_path = health_centres_path
        self.tracts = tracts_gdf
        self.TractID = TractID
        self.patient_id = patient_id

    def compute_matrix(self, cutoff=0.2):
        # Assign tract IDs
        self.patients_assignment = self.assign_tracts(self.origins, self.patient_id, "TractID_1")
        self.centres_assignment = self.assign_tracts(self.health_centres, "care_cen_code", "TractID_2")

        # Merge patient and tract data
        data = self.patient_data.merge(self.patients_assignment, on=self.patient_id, how="left")
        data = data[data["TractID_1"].notna()]
        data["TractID_1"] = data["TractID_1"].astype("Int64")
        data = data.merge(self.centres_assignment, on="care_cen_code", how="left")
        self.data = data.copy()

        data = self.calculate_visit_shares(data)
        
        #data = self.assign_tract_pairs(data)

        matrix = self.aggregate_visits_by_pair(data)
        matrix = self.normalise_and_score_matrix(matrix)

        # Get representative coordinates per tract
        self.origins = self.origins.merge(self.patients_assignment, on=self.patient_id, how="left")
        tract_coords = self.compute_tract_coordinates(
            origins_gdf=self.origins,
            tracts_gdf=self.tracts,
            cutoff=cutoff
            )

        # Map coordinates
        matrix = self.map_coordinates_to_matrix(matrix, tract_coords)
        self.matrix = matrix

    def load_patient_data(self):
        patient_path = Path(self.patient_data_path)
        locations_path = Path(self.locations_data_path)

        # Load tabular patient data
        if patient_path.suffix == ".csv":
            self.patient_data = pd.read_csv(self.patient_data_path)

        elif patient_path.suffix in [".parquet", ".gz", ".br", ".zstd"] or ".parquet" in patient_path.name:
            self.patient_data = pd.read_parquet(self.patient_data_path)

        else:
            raise ValueError(f"Unsupported file format for patient data: {self.patient_data_path}")

        # Load geospatial origins (locations)
        if locations_path.suffix in [".parquet", ".gz", ".br", ".zstd"] or ".parquet" in locations_path.name:
            self.origins = gpd.read_parquet(self.locations_data_path)
        else:
            self.origins = gpd.read_file(self.locations_data_path)

    def load_tracts(self):
        if self.tracts is not None:
            return
        elif self.tracts_path:
            self.tracts = gpd.read_file(self.tracts_path).drop(columns="pop", errors="ignore")
        else:
            raise ValueError("Either tract_gdf or tracts_path must be provided.")

    def load_health_centres(self):
        df = pd.read_excel(self.health_centres_path, sheet_name="cleaned")
        df = df.rename(columns={
            "Código Vigente": "care_cen_code",
            "Nombre Oficial": "centre_name",
            "Tipo Establecimiento (Unidad)": "facility_type",
            "LATITUD": "latitude",
            "LONGITUD": "longitude"
        })

        df = df[["care_cen_code", "centre_name", "facility_type", "latitude", "longitude"]]
        
        # Geocode health centres
        self.health_centres = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326"
        ).to_crs(32719)

        # Filter centres that exist in data
        self.health_centres = self.health_centres[
            self.health_centres["care_cen_code"].isin(self.patient_data["care_cen_code"])
        ]
        
    def load_all_data(self):
        self.load_tracts()
        self.load_patient_data()
        self.load_health_centres()

    def assign_tracts(self, geodf, geoid_field, target_field_name):
        intersection = gpd.sjoin(
        geodf, self.tracts[[self.TractID, "geometry"]],
        how="inner", predicate="intersects"
        )

        duplicate_mask = intersection[geoid_field].duplicated()

        if duplicate_mask.any():
            duplicates = intersection[duplicate_mask]
            nearest = gpd.sjoin_nearest(
                geodf[geodf[geoid_field].isin(duplicates[geoid_field].unique())],
                self.tracts[[self.TractID, "geometry"]],
                how="inner"
            )
            non_duplicates = intersection[~duplicate_mask]
            final = pd.concat([non_duplicates[[geoid_field, self.TractID]], nearest[[geoid_field, self.TractID]]])
        else:
            final = intersection[[geoid_field, self.TractID]]

        return final.rename(columns={self.TractID: target_field_name})

    def calculate_visit_shares(self, data):
        """
        Adds total_visits and visit_share columns to the data.
        """
        data = data.copy()
        data["total_visits"] = data.groupby(self.patient_id)["n_visits"].transform("sum")
        data["visit_share"] = data["n_visits"] / data["total_visits"]
        return data

    def assign_tract_pairs(self, data):
        """
        Creates undirected tract pairs (TractID_1, TractID_2) from TractID_1 and TractID_2,
        ensuring the lower ID is always TractID_1 for consistent grouping.

        Args:
            data (pd.DataFrame): DataFrame with 'TractID_1' and 'TractID_2' columns.

        Returns:
            pd.DataFrame: Original DataFrame with added 'TractID_1' and 'TractID_2' columns.
        """
        data = data.copy()

        tract_pairs = np.sort(data[["TractID_1", "TractID_2"]].values, axis=1)

        data[["TractID_1", "TractID_2"]] = pd.DataFrame(tract_pairs, index=data.index).astype("Int64")

        return data


    def aggregate_visits_by_pair(self, data):
        """
        Aggregates total visits and visit share by undirected tract pairs.
        """
        grouped = (
            data.groupby(["TractID_1", "TractID_2"])[["n_visits", "visit_share"]]
            .sum()
            .reset_index()
        )
        return grouped

    def normalise_and_score_matrix(self, matrix):
        """
        Normalises 'n_visits' and 'visit_share' columns and computes a combined score.
        """
        matrix = matrix.copy()
        scaler = MinMaxScaler()
        matrix[["visits_norm", "visit_share_norm"]] = scaler.fit_transform(matrix[["n_visits", "visit_share"]])
        matrix["combined_score"] = matrix[["visits_norm", "visit_share_norm"]].mean(axis=1)
        return matrix

    def compute_tract_coordinates(self, origins_gdf, tracts_gdf, cutoff=0.2, id_col='TractID', origin_id_col='TractID_1'):
        """
        Computes one representative coordinate per tract.

        For tracts with area > `cutoff` km² and with origin points inside, uses average of points within the tract,
        clipped to the tract geometry. For smaller tracts or those without points, uses representative_point().

        Args:
            origins_gdf (GeoDataFrame): Points with patient origins (must include origin_id_col).
            tracts_gdf (GeoDataFrame): Polygons representing tracts with unique IDs (id_col).
            cutoff (float): Area threshold in square kilometres.
            id_col (str): Name of the tract ID column in tracts_gdf.
            origin_id_col (str): Name of the tract ID column in origins_gdf.

        Returns:
            GeoDataFrame: with columns [id_col, 'longitude', 'latitude'] and geometry as points.
        """
        # Copy and project to a metric CRS for area calculation and coordinate accuracy
        tracts = tracts_gdf.to_crs(epsg=32719).copy()
        origins = origins_gdf.to_crs(epsg=32719).copy()

        # Calculate tract area in km²
        tracts['area_km2'] = tracts.geometry.area / 1e6

        # Prepare a list for results
        records = []

        # Create a dict for quick tract geometry lookup
        tracts_geom = tracts.set_index(id_col).geometry

        # Group origin points by their tract ID for efficient access
        grouped_origins = origins.groupby(origin_id_col)

        for tract_id, tract_geom in tracts_geom.items():
            area = tracts.loc[tracts[id_col] == tract_id, 'area_km2'].iloc[0]

            # Get points belonging to this tract
            points_in_tract = grouped_origins.get_group(tract_id) if tract_id in grouped_origins.groups else None

            if area > cutoff and points_in_tract is not None and not points_in_tract.empty:
                # Calculate mean x and y of points
                mean_x = points_in_tract.geometry.x.mean()
                mean_y = points_in_tract.geometry.y.mean()
                avg_point = gpd.points_from_xy([mean_x], [mean_y])[0]

                # If mean point lies outside the tract polygon, fallback to representative_point
                if not tract_geom.contains(avg_point):
                    avg_point = tract_geom.representative_point()
            else:
                # For smaller or empty tracts, use representative_point
                avg_point = tract_geom.representative_point()

            records.append({
                id_col: tract_id,
                'geometry': avg_point
            })

        # Create GeoDataFrame with original CRS (projected)
        coords_gdf = gpd.GeoDataFrame(records, crs=tracts.crs)

        coords_gdf['longitude'] = coords_gdf.geometry.x
        coords_gdf['latitude'] = coords_gdf.geometry.y

        return coords_gdf[[id_col, 'longitude', 'latitude']]

    def map_coordinates_to_matrix(self, matrix, tract_coords):
        """
        Adds coordinates (lat/lon) to matrix based on TractID_1 and TractID_2.

        Args:
            matrix: DataFrame with columns TractID_1, TractID_2
            tract_coords: DataFrame with columns TractID, longitude, latitude

        Returns:
            matrix with added columns: lon_1, lat_1, lon_2, lat_2
        """
        lon_map = dict(zip(tract_coords[self.TractID], tract_coords["longitude"]))
        lat_map = dict(zip(tract_coords[self.TractID], tract_coords["latitude"]))

        matrix["lon_1"] = matrix["TractID_1"].map(lon_map)
        matrix["lat_1"] = matrix["TractID_1"].map(lat_map)
        matrix["lon_2"] = matrix["TractID_2"].map(lon_map)
        matrix["lat_2"] = matrix["TractID_2"].map(lat_map)

        return matrix

    def export_matrix(self, out_path):
        self.matrix.to_csv(out_path, index=False)

    def create_flow_lines(self):
        """
        Create a GeoDataFrame of LineStrings representing flows 
        between origin and destination tracts
        """
        flow_lines = []

        for _, row in self.matrix.iterrows():
            # Get coordinates
            origin_coords = (row["lon_1"], row["lat_1"])
            dest_coords = (row["lon_2"], row["lat_2"])

            # Check for missing coordinates
            if None not in origin_coords and None not in dest_coords:
                line = LineString([origin_coords, dest_coords])
                flow_lines.append({
                    "geometry": line,
                    "TractID_1": row["TractID_1"],
                    "TractID_2": row["TractID_2"],
                    "n_visits": row["n_visits"],
                    "visit_share": row["visit_share"],
                    "combined_score": row["combined_score"]
                })

        # Create GeoDataFrame
        flow_gdf = gpd.GeoDataFrame(flow_lines, crs="EPSG:32719")

        return flow_gdf
    
    def get_top_centre_per_community(self, community_lookup_df, community_col="community", weight_col="n_visits"):
        """
        Returns the health centre with the most visits for each community.

        Args:
            community_lookup_df (pd.DataFrame): DataFrame with 'TractID' and 'community' columns.

        Returns:
            pd.DataFrame: DataFrame with 'community' and the corresponding 'care_cen_code' with most visits.
        """
        if not hasattr(self, "data"):
            raise AttributeError("Patient-centre visit data not found. Run `compute_matrix()` first.")

        data = self.data.copy()
        data["TractID_1"] = data["TractID_1"].astype(str)

        # Merge community information
        data = data.merge(community_lookup_df, left_on="TractID_1", right_on=self.TractID, how="left")

        # Aggregate visits per community and centre
        grouped = (
            data.groupby([community_col, "care_cen_code"])[weight_col]
            .sum()
            .reset_index()
        )

        # Get top centre per community
        idx_max = grouped.groupby(community_col)[weight_col].idxmax()
        top_centre_per_community = grouped.loc[idx_max].copy().drop(columns=weight_col)

        return top_centre_per_community
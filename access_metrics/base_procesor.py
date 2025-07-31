import pandas as pd

class BaseProcessor:
    """
    Processes patient-level consultation data by aggregating public and private visits.

    This class loads, aggregates, and combines data from public and private consultations,
    generating a unified long-format DataFrame with total visits by visit type and group
    identifiers.

    Attributes:
        data (pd.DataFrame): Input public consultation data.
        patient_id (str): Column name for patient identifier.
        patient_data_mle_path (str): File path to private consultation data (Parquet format).
        public_visits_col (str): Column name for public visit counts.
        private_visits_col (str): Column name for private visit counts.
        total_visits_col (str): Column name for total visit counts.
        geo_id (str): Column name for geographic unit identifier.
        group_cols (list of str): Columns used for grouping during aggregation.
    """
    def __init__(self,
                 data=None,
                 patient_id="id",
                 patient_data_mle_path=None,
                 public_visits_col="n_visits",
                 private_visits_col="n_visits_mle",
                 total_visits_col="Total",
                 geo_id="TractID_origin",
                 group_cols=["sex", "age_group"]):
        self.data = data
        self.patient_id = patient_id
        self.patient_data_mle_path = patient_data_mle_path
        self.public_visits_col = public_visits_col
        self.private_visits_col = private_visits_col
        self.total_visits_col = total_visits_col
        self.geo_id = geo_id
        self.group_cols = [self.patient_id] + group_cols + [self.geo_id]
        
    def _load_public_consultations(self):
        data_public = self.data.groupby(self.group_cols)[self.public_visits_col].sum().reset_index()
        self.data_public = data_public
        
    def _load_private_consultations(self):
        data_mle = pd.read_parquet(self.patient_data_mle_path)
        data_mle = data_mle.groupby(self.patient_id)[self.private_visits_col].sum().reset_index()
        self.data_mle = data_mle

    def _combine_all_data(self):
        total = self.data_public.merge(self.data_mle, on=self.patient_id, how="left")
        total[[self.public_visits_col, self.private_visits_col]] = total[[self.public_visits_col, self.private_visits_col]].fillna(0)
        total[[self.public_visits_col, self.private_visits_col]] = total[[self.public_visits_col, self.private_visits_col]].astype(int)
        total[self.total_visits_col] = total[self.public_visits_col] + total[self.private_visits_col]

        total_long = (
            pd.melt(
                total,
                id_vars=self.group_cols,
                value_vars=[self.public_visits_col, self.private_visits_col, self.total_visits_col],
                var_name="visit_type",
                value_name="visits"
            )
            .reset_index(drop=True)
        )
        
        replace_dict = {
        self.public_visits_col: "Public",
        self.private_visits_col: "Private"
        }

        total_long["visit_type"] = total_long["visit_type"].replace(replace_dict)

        #total_long = total_long.reset_index()
        self.total = total_long
    
    def _get_all_data(self):
        """"""
        self._load_public_consultations()
        self._load_private_consultations()
        self._combine_all_data()
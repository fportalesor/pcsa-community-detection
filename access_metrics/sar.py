
class SARCalculator:
    """
    Calculates the Standardised Access Ratio (SAR) and Relative Access Ratio (RAR)
    per geographic area using direct standardisation.

    This class estimates observed and expected access to services (e.g., health care) 
    based on group-specific reference rates, and computes SAR and RAR values per area 
    and visit type.

    Args:
        data (pd.DataFrame): Input dataset containing individual-level or grouped visit data.
        patient_id (str): Column name identifying unique individuals or patients.
        visits_type (str): Column indicating the type of service (e.g., 'Public', 'Private').
        visits_value (str): Column containing the number of visits or usage count.
        area_id (str): Column identifying the geographic area (e.g., tract or zone).
        group_cols (list of str): Demographic or other grouping columns for stratification 
            (e.g., sex, age group).
    """

    def __init__(self, 
                 data=None,
                 patient_id="id",
                 visits_type="visits_type",
                 visits_value="visits",
                 area_id="TractID_origin",
                 group_cols=["sex", "age_group"]):
        self.data = data
        self.patient_id = patient_id
        self.visits_type = visits_type
        self.visits_value = visits_value
        self.area_id = area_id
        self.group_cols = group_cols + [visits_type]
        
        self.data['has_accessed'] = self.data[self.visits_value] > 0

    def _process(self):
        self._get_overall_ref_rates()
        self._get_ref_rates_by_groups()
        self._get_expected_access()
        self._calculate_SAR()
        self._calculate_SRAR()

    def _get_overall_ref_rates(self):

        ref_rates = self.data.groupby(self.visits_type).agg(
        accessed=('has_accessed', 'sum'),
        total_population=(self.patient_id, 'nunique')
        ).reset_index()

        ref_rates['ref_rate'] = (ref_rates['accessed'] /
                                      ref_rates['total_population'])

        self.ref_rates_overall = ref_rates

    def _get_ref_rates_by_groups(self):

        ref_rates = self.data.groupby(self.group_cols).agg(
        accessed=('has_accessed', 'sum'),
        total_population=(self.patient_id, 'nunique')
        ).reset_index()

        ref_rates['ref_rate'] = (ref_rates['accessed'] /
                                      ref_rates['total_population'])

        self.ref_rates_groups = ref_rates
    
    def _get_expected_access(self):
        exp = self.data.groupby([self.area_id]  + self.group_cols).agg(
            accessed=('has_accessed', 'sum'),
            population=(self.patient_id, 'nunique')
            ).reset_index()
        
        exp["observed_rate"] = exp['accessed'] / exp['population']

        exp = exp.merge(self.ref_rates_groups[self.group_cols + ['total_population']],
                  on=self.group_cols, how='left')

        exp['expected'] = exp['observed_rate'] * exp['total_population']
        self.exp = exp
    
    def _calculate_SAR(self):
        area_rates = self.exp.groupby([self.area_id]+[self.visits_type]).agg(
            total_expected=('expected', 'sum'),
            total_population=('total_population', 'sum')
        ).reset_index()

        area_rates['SAR'] = area_rates['total_expected'] / area_rates['total_population']

        self.area_rates = area_rates

    def _calculate_SRAR(self):

        self.area_rates = self.area_rates.merge(self.ref_rates_overall[["ref_rate", self.visits_type]], 
                                                on=self.visits_type, how="left")
        
        self.area_rates["SRAR"] = self.area_rates["SAR"] / self.area_rates["ref_rate"]
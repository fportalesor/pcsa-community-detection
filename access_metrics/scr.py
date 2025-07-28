
class SCRCalculator:
    """
    Calculates the Standardised Consulation Ratio (SCR) and Relative Consulation Ratio (RCR) 
    per geographic area using direct standardisation.

    This class estimates experved and expected consultations to services (e.g., health care) 
    based on group-specific reference rates, and computes SCR and RCR values per area 
    and visit type.

    Args:
        data (pd.DataFrame): Input dataset containing individual-level or grouped visit data.
        patient_id (str): Column name identifying unique individuals or patients.
        visits_type (str): Column indicating the type of service (e.g., 'Public', 'Private').
        visits_value (str): Column containing the number of visits or usage count.
        area_id (str): Column identifying the geographic area (e.g., tract or zone).
        group_cols (list of str): Demographic or other grouping columns for stratification 
            (e.g., sex, age group).
        adjusted_rate (bool): Whether adjust rates using just population 
            that has accessed as denominator.
    """

    def __init__(self, 
                 data=None,
                 patient_id="id",
                 visits_type="visits_type",
                 visits_value="visits",
                 area_id="TractID_origin",
                 group_cols=["sex", "age_group"],
                 adjusted_rate=True):
        self.data = data
        self.patient_id = patient_id
        self.visits_type = visits_type
        self.visits_value = visits_value
        self.area_id = area_id
        self.group_cols = group_cols + [visits_type]
        self.adjusted_rate = adjusted_rate
        
        self.data['has_accessed'] = self.data[self.visits_value] > 0

    def _process(self):
        self._get_overall_ref_rates()
        self._get_ref_rates_by_groups()
        self._get_expected_consultations()
        self._calculate_SCR()
        self._calculate_RCR()

    def _get_overall_ref_rates(self):

        if self.adjusted_rate:
            ref_rates = self.data.groupby(self.visits_type).agg(
            total_consultations=(self.visits_value, 'sum'),
            total_population=('has_accessed', 'sum')
            ).reset_index()
        else:
            ref_rates = self.data.groupby(self.visits_type).agg(
            total_consultations=(self.visits_value, 'sum'),
            total_population=(self.patient_id, 'nunique')
            ).reset_index()


        ref_rates['ref_rate'] = (ref_rates['total_consultations'] /
                                      ref_rates['total_population'])

        self.ref_rates_overall = ref_rates

    def _get_ref_rates_by_groups(self):

        if self.adjusted_rate:
            ref_rates = self.data.groupby(self.group_cols).agg(
            total_consultations=(self.visits_value, 'sum'),
            total_population=('has_accessed', 'sum')
            ).reset_index()
        else:
            ref_rates = self.data.groupby(self.group_cols).agg(
            total_consultations=(self.visits_value, 'sum'),
            total_population=(self.patient_id, 'nunique')
            ).reset_index()

        ref_rates['ref_rate'] = (ref_rates['total_consultations'] /
                                    ref_rates['total_population'])

        self.ref_rates_groups = ref_rates
    
    def _get_expected_consultations(self):
        if self.adjusted_rate:
            exp = self.data.groupby([self.area_id]  + self.group_cols).agg(
                consultations=(self.visits_value, 'sum'),
                population=('has_accessed', 'sum')
                ).reset_index()
        else:    
            exp = self.data.groupby([self.area_id]  + self.group_cols).agg(
                consultations=(self.visits_value, 'sum'),
                population=(self.patient_id, 'nunique')
                ).reset_index()
        
        exp["observed_rate"] = exp['consultations'] / exp['population']

        exp = exp.merge(self.ref_rates_groups[self.group_cols + ['total_population']],
                  on=self.group_cols, how='left')

        exp['expected'] = exp['observed_rate'] * exp['total_population']

        self.exp = exp
    
    def _calculate_SCR(self):
        area_rates = self.exp.groupby([self.area_id]+[self.visits_type]).agg(
            total_expected=('expected', 'sum'),
            total_population=('total_population', 'sum')
        ).reset_index()

        area_rates['SCR'] = area_rates['total_expected'] / area_rates['total_population']
                      
        self.area_rates = area_rates

    def _calculate_RCR(self):

        self.area_rates = self.area_rates.merge(self.ref_rates_overall[["ref_rate", self.visits_type]], 
                                                on=self.visits_type, how="left")
        
        self.area_rates["RCR"] = self.area_rates["SCR"] / self.area_rates["ref_rate"]
import numpy as np

class AccessCategoriser:
    """
    Categorises geographic areas based on access scores to public, private, and total services.

    This class reshapes access metric data into wide format and assigns each area a category
    based on predefined conditions that compare relative access ratios across service types,
    where a value of 1 indicates parity with the overall (or reference) access level.

    Attributes:
        data (pd.DataFrame): Input DataFrame in long format with access scores.
        area_id (str): Column name for the unique area identifier.
        visit_type (str): Column name indicating the type of service (e.g., 'Public', 'Private').
        public_vis_col (str): Column name for public service access score in wide format.
        private_vis_col (str): Column name for private service access score in wide format.
        total_vis_col (str): Column name for total service access score in wide format.
        measure (str): Column name of the access score to categorise.
    """
    def __init__(self, 
                 data,
                 area_id='TractID',
                 visit_type='visit_type',
                 public_vis_col='Public',
                 private_vis_col='Private',
                 total_vis_col='Total',
                 measure='SRAR'):
        self.data = data
        self.area_id = area_id
        self.visit_type = visit_type
        self.public_vis_col = public_vis_col
        self.private_vis_col = private_vis_col
        self.total_vis_col = total_vis_col
        self.measure = measure


    def _wide_format(self):
        data_wide = self.data.pivot(index=self.area_id, columns=self.visit_type, values=self.measure).reset_index()
        data_wide.columns.name = None # Remove the name of the column index
        self.data_wide = data_wide

    def _categorise(self):
        conditions = [
            (self.data_wide[self.public_vis_col] > 1.0) & 
            (self.data_wide[self.private_vis_col] > 1.0),

            (self.data_wide[self.public_vis_col] < 1.0) & 
            (self.data_wide[self.private_vis_col] < 1.0),

            (self.data_wide[self.public_vis_col] > 1.0) & 
            (self.data_wide[self.private_vis_col] < 1.0) & 
            (self.data_wide[self.total_vis_col] > 1.0),

            (self.data_wide[self.public_vis_col] > 1.0) & 
            (self.data_wide[self.private_vis_col] < 1.0) & 
            (self.data_wide[self.total_vis_col] < 1.0),

            (self.data_wide[self.public_vis_col] < 1.0) & 
            (self.data_wide[self.private_vis_col] > 1.0) & 
            (self.data_wide[self.total_vis_col] > 1.0),

            (self.data_wide[self.public_vis_col] < 1.0) & 
            (self.data_wide[self.private_vis_col] > 1.0) & 
            (self.data_wide[self.total_vis_col] < 1.0)
        ]

        choices = ['Equitable Access', 
                   'Limited Access',
                   'Compensated through Public Access',
                   'Insufficient Public Compensation', 
                   'Compensated through Private Access',
                   'Insufficient Private Compensation']

        self.data_wide['Category'] = np.select(conditions, choices, default='Unknown')
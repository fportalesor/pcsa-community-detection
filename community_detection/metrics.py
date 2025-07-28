class CommunityMetrics:
    """
    Provides metrics for evaluating community aggregations, such as the Localisation Index (LI).
    """

    def __init__(self, matrix, tracts, community_col="community"):
        """
        Initialise the CommunityMetrics class.

        Args:
            matrix (pd.DataFrame): Flow matrix containing columns 'TractID_1', 'TractID_2', and a weight column (e.g., visits).
            tracts (gpd.GeoDataFrame): GeoDataFrame of tracts, typically including community assignments.
            community_col (str): Column name in `tracts` used for community labels (default is 'community').
        """
        self.matrix = matrix
        self.tracts = tracts
        self.community_col = community_col

        # Ensure consistent ID formats
        self.matrix[["TractID_1", "TractID_2"]] = self.matrix[["TractID_1", "TractID_2"]].astype(str)

    def calculate_localisation_index(self, weigth_col="n_visits", community_assignment=None):
        """
        Calculates the Localisation Index (LI), which quantifies how self-contained communities are
        based on flows (e.g., visits) within and between them.

        Args:
            weigth_col (str): Column in the flow matrix representing the magnitude of flow (default is 'n_visits').
            community_assignment (dict): Mapping from TractID to community label.

        Returns:
            pd.DataFrame: DataFrame with localisation index per community, including:
                - 'community': Community ID
                - 'D_c': Total outgoing flow from the community
                - 'D_cc': Total intra-community flow
                - 'LI': Localisation index (D_cc / D_c)
        """
        self.matrix["origin_comm"] = self.matrix["TractID_1"].map(community_assignment)
        self.matrix["dest_comm"] = self.matrix["TractID_2"].map(community_assignment)

        # Total visits from origin community
        total_outgoing = self.matrix.groupby("origin_comm")[weigth_col].sum().rename("D_c").reset_index()

        # Visits within same community
        within_community = self.matrix[self.matrix["origin_comm"] == self.matrix["dest_comm"]]
        within_outgoing = within_community.groupby("origin_comm")[weigth_col].sum().rename("D_cc").reset_index()

        # Combine and calculate LI
        li_df = total_outgoing.merge(within_outgoing, on="origin_comm", how="left").fillna(0)
        li_df["LI"] = li_df["D_cc"] / li_df["D_c"]

        li_df = li_df.rename(columns={"origin_comm": "community"})
        li_df = li_df[["community", "D_c", "D_cc", "LI"]]

        return li_df

    @staticmethod
    def summarise_localisation_index(li_df, exclude_zero_flows=False, groupby_cols=None):
        """
        Generates summary statistics for the Localisation Index (LI) across communities.

        Args:
            li_df (pd.DataFrame): Output DataFrame from `calculate_localisation_index`.
            exclude_zero_flows (bool): If True, excludes communities with no intra-community flow.
            groupby_cols (list, optional): Columns to group by before aggregating (e.g., method, region).

        Returns:
            pd.DataFrame: Summary statistics including mean, median, min, max, and standard deviation of LI,
                          total and within-community flows, and number of communities.
        """
        if exclude_zero_flows:
            li_df = li_df[li_df['D_cc'] > 0]

        summary_stats = li_df.groupby(groupby_cols).agg(
            mean_LI=('LI', 'mean'),
            median_LI=('LI', 'median'),
            min_LI=('LI', 'min'),
            max_LI=('LI', 'max'),
            std_LI=('LI', 'std'),
            n_communities=('LI', 'count'),
            total_visits=('D_c', 'sum'),
            within_community_visits=('D_cc', 'sum')
        ).reset_index()

        summary_stats['exclude_zero_flows'] = exclude_zero_flows

        return summary_stats
from polygon_processors import PolygonProcessor

class PolygonDissolver(PolygonProcessor):
    """
    Dissolve polygons based on assigned Community definition.
    """

    def __init__(self, input_data=None):
        self.data = input_data

    def process_community_outcomes(
        self,
        tract_id="TractID",
        communities=None,
        community_id="community",
        fill_holes=True,
        remove_dangles=True,
        verbose=True
    ):
        """
        Processes input polygon data and dissolves them into Primary Care Service Areas
            (PCSAs) using a Community detection Output.

        Args:
            tract_id (str): Column name for the dissolve grouping identifier (default: 'TractID').
            communities (Dataframe): 
            community_id (str):
            fill_holes (bool): Whether to fill small internal holes in the polygons.
            remove_dangles (bool): Whether to remove inward-facing dangles in polygon boundaries.
            verbose (bool): If True, prints intermediate information during processing.

        Returns:
            GeoDataFrame: Dissolved polygons
        """

        # Merge Tracts with Community assignments
        merged_tracts = self.data.merge(communities, on=tract_id, how="left")
        
        def most_frequent(x):
            return x.mode().iloc[0] if not x.mode().empty else x.iloc[0]

        pcsa = merged_tracts.dissolve(by="community",aggfunc={
                "commune_id": most_frequent,
                "commune": most_frequent,
                "pop": "sum",
                "pop_high": "sum",
                "pop_middle": "sum",
                "pop_low": "sum"
            }
        ).reset_index()

        pcsa = pcsa[
            ["commune_id", "commune", community_id,
             "pop", "pop_high", "pop_middle", "pop_low", "geometry"]
        ]
            
        if fill_holes:
            if verbose:
                print("\nFixing internal holes...")
            pcsa.geometry = (
                pcsa.geometry
                .apply(lambda geom: self.fill_holes(geom, sizelim=10))
            )
        
        if remove_dangles:
            if verbose:
                print("\nRemoving dangles...")
            pcsa.geometry = pcsa.geometry.apply(
                self.remove_dangles
            )

        # Check and report multipart geometries
        _, duplicates = self.identify_multipart_polygons(pcsa, community_id, keep_largest=False)
        if not duplicates.empty and verbose:
            print(f"Warning: {len(duplicates)} multipart polygons still remain after processing.")
        
        return pcsa
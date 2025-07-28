import networkx as nx
import pandas as pd

class NXGraphConstructor:
    def __init__(self, matrix_df, weights_col="n_visits"):
        self.matrix = matrix_df
        self.graph = None
        self.weights_col = weights_col

    def build_graph(self):
        # Extract origins and destinations
        origins = self.matrix[["TractID_1", "lon_1", "lat_1"]].dropna().copy()
        origins.columns = ["TractID", "lon", "lat"]

        destinations = self.matrix[["TractID_2", "lon_2", "lat_2"]].dropna().copy()
        destinations.columns = ["TractID", "lon", "lat"]

        # Combine and drop duplicates to get unique nodes
        all_coords_df = pd.concat([origins, destinations]).drop_duplicates()
        all_coords_df["TractID"] = all_coords_df["TractID"].astype(str)

        # Create a dict for coordinates lookup
        coord_map = all_coords_df.set_index("TractID")[["lon", "lat"]].to_dict("index")

        # Initialise an undirected graph
        G = nx.Graph()

        # Add nodes with attributes (TractID, lon, lat)
        for tid, coords in coord_map.items():
            G.add_node(tid, TractID=tid, lon=coords["lon"], lat=coords["lat"])

        # Add edges with weights
        for _, row in self.matrix.iterrows():
            origin = str(row["TractID_1"])
            destination = str(row["TractID_2"])
            weight = row[self.weights_col]

            # Add edge only if both nodes exist
            if origin in G and destination in G:
                G.add_edge(origin, destination, weight=weight)

        self.graph = G

    def get_graph(self):
        return self.graph
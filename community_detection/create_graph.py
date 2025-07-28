import igraph as ig
import pandas as pd

class GraphConstructor:
    def __init__(self, matrix_df, weights_col="n_visits"):
        self.matrix = matrix_df
        self.graph = None
        self.weights_col = weights_col

    def build_graph(self):
        # Get unique node IDs from both origin and destination
        origins = self.matrix[["TractID_1", "lon_1", "lat_1"]].dropna().copy()
        origins.columns = ["TractID", "lon", "lat"]

        destinations = self.matrix[["TractID_2", "lon_2", "lat_2"]].dropna().copy()
        destinations.columns = ["TractID", "lon", "lat"]

        all_coords_df = pd.concat([origins, destinations]).drop_duplicates()
        all_coords_df["TractID"] = all_coords_df["TractID"].astype(str)

        # Unique node IDs and mapping
        all_nodes = all_coords_df["TractID"].unique()
        self.id_to_index = {tid: i for i, tid in enumerate(all_nodes)}
        self.index_to_id = {i: tid for tid, i in self.id_to_index.items()}

        # Build edges
        edges = [
            (self.id_to_index[str(row["TractID_1"])],
             self.id_to_index[str(row["TractID_2"])])
            for _, row in self.matrix.iterrows()
        ]
        weights = self.matrix[self.weights_col].tolist()

        # Create graph
        self.graph = ig.Graph(edges=edges, edge_attrs={"weight": weights}, directed=False)

        # Add vertex attributes
        self.graph.vs["TractID"] = [self.index_to_id[i] for i in range(len(self.index_to_id))]

        # Prepare coordinate map
        coord_map = all_coords_df.set_index("TractID")[["lon", "lat"]].to_dict("index")

        # Add lon and lat attributes
        self.graph.vs["lon"] = [coord_map[tid]["lon"] for tid in self.graph.vs["TractID"]]
        self.graph.vs["lat"] = [coord_map[tid]["lat"] for tid in self.graph.vs["TractID"]]

    def get_graph(self):
        return self.graph

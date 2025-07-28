from libpysal.weights import Queen
from collections import defaultdict
from infomap import Infomap
import networkx as nx
from .enforcement_strategies import EnforcementStrategies

class SpatialContiguityEnforcer:
    """Ensures spatial contiguity in community detection results through tract reassignment.
    
    Supports two strategies: minimal score impact (modularity/codelength) or strongest connections.
    Requires geographic data (GeoDataFrame) and initial community assignments as input.
    """
    def __init__(self, gdf, community_dict, score=None, g_ig=None, tract_id_col='TractID', verbose=True):
        self.gdf = gdf.copy()
        self.g_ig = g_ig
        self.tract_id_col = tract_id_col
        self.verbose = verbose
        self.min_impact_score = float('-inf')

        self._build_mappings()
        self._build_adjacency()

        self.community_dict = {}
        self.missing_tracts = set()
        for tract, comm in community_dict.items():
            tract_str = str(tract)
            if tract_str in self.tract_to_idx:
                self.community_dict[tract_str] = comm
            else:
                self.missing_tracts.add(tract_str)

        if self.missing_tracts and self.verbose:
            print(f"Warning: {len(self.missing_tracts)} tracts from community results not found in spatial data")

        self.original_community_dict = self.community_dict.copy()
        self.original_score = score
        self._update_adjacency()
        self.strategies = EnforcementStrategies(self)

    def enforce_contiguity(self, score_metric='modularity', max_iterations=100, strategy='min_impact_score'):
        self.min_impact_score = self.original_score

        for iteration in range(max_iterations):
            self._update_adjacency()
            disconnected = self._find_disconnected_nodes()
            if not disconnected:
                if self.verbose:
                    print(f"Contiguity achieved after {iteration} iterations")
                break

            self._process_disconnected_tracts(disconnected, score_metric, strategy, iteration)

            self._update_adjacency()
            current_disconnected = self._find_disconnected_nodes()
            if self.verbose:
                print(f"Remaining disconnected after iteration {iteration}: {len(current_disconnected)}")

        self._validate_contiguity()
        self._calculate_final_score(strategy, score_metric)
        return self.community_dict

    def _build_mappings(self):
        self.tract_to_idx = {str(tract): idx for idx, tract in enumerate(self.gdf[self.tract_id_col])}
        self.idx_to_tract = {idx: tract for tract, idx in self.tract_to_idx.items()}

        if self.g_ig is not None:
            self.graph_tract_to_idx = {str(v[self.tract_id_col]): v.index for v in self.g_ig.vs}
            self.graph_idx_to_tract = {v.index: str(v[self.tract_id_col]) for v in self.g_ig.vs}

    def _build_adjacency(self):
        self.queen = Queen.from_dataframe(self.gdf, ids=self.gdf.index.tolist())
        self.adjacency_dict = self.queen.neighbors

        self.tract_neighbors = defaultdict(set)
        for idx, neighbors in self.adjacency_dict.items():
            tract = self.idx_to_tract[idx]
            self.tract_neighbors[tract] = {self.idx_to_tract[n] for n in neighbors}

    def _update_adjacency(self):
        self.filtered_adjacency = defaultdict(set)
        for tract, neighbors in self.tract_neighbors.items():
            if tract not in self.community_dict:
                continue
            current_comm = self.community_dict[tract]
            self.filtered_adjacency[tract] = {
                n for n in neighbors
                if n in self.community_dict and self.community_dict[n] == current_comm
            }

    def _find_disconnected_nodes(self):
        G = nx.Graph()
        for tract, neighbors in self.filtered_adjacency.items():
            if tract not in G:
                G.add_node(tract)
            for neighbor in neighbors:
                G.add_edge(tract, neighbor)

        disconnected = []
        community_groups = defaultdict(list)
        for tract, comm in self.community_dict.items():
            community_groups[comm].append(tract)

        for comm, tracts_in_comm in community_groups.items():
            subgraph_nodes = set(tracts_in_comm)
            subgraph = G.subgraph(subgraph_nodes)

            # Detect Orphan nodes
            if subgraph.number_of_edges() == 0:
                disconnected.extend(subgraph_nodes)
                continue

            components = list(nx.connected_components(subgraph))
            if len(components) <= 1:
                continue
            
            # Detect Enclaved/Enclosed nodes
            largest_component = max(components, key=len)
            for comp in components:
                if comp != largest_component:
                    disconnected.extend(comp)

        return disconnected

    def _get_adjacent_communities(self, tract):
        adjacent_comms = set()
        for neighbor in self.tract_neighbors.get(tract, []):
            neighbor_comm = self.community_dict.get(neighbor)
            if neighbor_comm is not None and neighbor_comm != self.community_dict[tract]:
                adjacent_comms.add(neighbor_comm)
        return adjacent_comms

    def _reassign_tract(self, tract, new_comm):
        self.community_dict[tract] = new_comm
        if self.g_ig is not None and tract in self.graph_tract_to_idx:
            vertex_idx = self.graph_tract_to_idx[tract]
            self.g_ig.vs[vertex_idx]['community'] = new_comm

    def _validate_contiguity(self):
        self._update_adjacency()
        remaining_disconnected = self._find_disconnected_nodes()
        if remaining_disconnected and self.verbose:
            print(f"Validation failed: {len(remaining_disconnected)} tracts remain disconnected")
            for tract in remaining_disconnected[:5]:
                print(f"Disconnected tract {tract} in community {self.community_dict[tract]}")
                print("Adjacent tracts:", self.tract_neighbors.get(tract, set()))
                print("Their communities:", {
                    n: self.community_dict.get(n)
                    for n in self.tract_neighbors.get(tract, set())
                })
        return len(remaining_disconnected) == 0

    def _process_disconnected_tracts(self, disconnected, score_metric, strategy, iteration):
        if self.verbose:
            print(f"Iteration {iteration}: Processing {len(disconnected)} disconnected tracts")

        for tract in disconnected:
            adjacent_comms = self._get_adjacent_communities(tract)
            if not adjacent_comms:
                if self.verbose:
                    print(f"Tract {tract} has no adjacent communities - cannot reassign")
                continue

            self._handle_tract_reassignment(tract, adjacent_comms, score_metric, strategy)

    def _handle_tract_reassignment(self, tract, adjacent_comms, score_metric, strategy):
        if len(adjacent_comms) == 1:
            new_comm = adjacent_comms.pop()
            self._reassign_tract(tract, new_comm)
            if self.verbose:
                print(f"Reassigned tract {tract} to the only adjacent community: {new_comm}")
        else:
            if strategy == 'min_impact_score':
                selected_comm, score = self.strategies.find_min_impact_score(
                    tract, adjacent_comms, score_metric, baseline_score=self.original_score
                )
                if (score_metric == 'modularity' and score >= self.min_impact_score) or \
                (score_metric != 'modularity' and abs(score - self.original_score) < abs(self.min_impact_score - self.original_score)):
                    self.min_impact_score = score

                if self.verbose:
                    print(f"Reassigned tract {tract} to community {selected_comm} with impact score {score}")
            elif strategy == 'strongest_connection':
                selected_comm = self.strategies.find_strongest_connection(tract, adjacent_comms)
                self._reassign_tract(tract, selected_comm)
                if self.verbose:
                    print(f"Reassigned tract {tract} to strongest-connected community {selected_comm}")
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

    def _calculate_final_score(self, strategy, score_metric):
        if strategy == 'strongest_connection' and self.g_ig is not None:
            membership = [
                self.community_dict.get(self.graph_idx_to_tract[v.index])
                for v in self.g_ig.vs
            ]
            if score_metric == 'modularity':
                self.min_impact_score = self.g_ig.modularity(membership, weights='weight')
            elif score_metric == 'codelength':
                im = Infomap("--two-level --silent")
                for edge in self.g_ig.es:
                    im.add_link(edge.source, edge.target, edge["weight"])
                im.run()
                self.min_impact_score = im.codelength

    # Reporting methods
    def get_final_score(self):
        return self.min_impact_score

    def get_contiguity_report(self):
        is_contiguous = self._validate_contiguity()
        report = {
            'total_tracts': len(self.community_dict),
            'missing_tracts': len(self.missing_tracts),
            'tracts_reassigned': sum(
                1 for tract in self.community_dict
                if self.community_dict[tract] != self.original_community_dict.get(tract)
            ),
            'is_fully_contiguous': is_contiguous,
            'remaining_disconnected': len(self._find_disconnected_nodes())
        }
        return report
from collections import defaultdict
from infomap import Infomap

class EnforcementStrategies:
    """Implements different strategies for enforcing spatial contiguity during community reassignment.
    """
    def __init__(self, enforcer):
        self.enforcer = enforcer

    def find_strongest_connection(self, tract, candidate_comms):
        """Determines the best community based on edge weights in the interaction graph.

        Args:
            tract (str): The tract ID being reassigned.
            candidate_comms (set): Set of candidate community IDs to consider.

        Returns:
            str: The community ID with the strongest connections to the tract.

        Note:
            Falls back to first candidate community if no edge weights are found.
        """
        if self.enforcer.g_ig is None or tract not in self.enforcer.graph_tract_to_idx:
            return list(candidate_comms)[0]

        tract_idx = self.enforcer.graph_tract_to_idx[tract]
        edge_weights = defaultdict(float)

        for neighbor_idx in self.enforcer.g_ig.neighbors(tract_idx):
            neighbor_tract = self.enforcer.graph_idx_to_tract[neighbor_idx]
            neighbor_comm = self.enforcer.community_dict.get(neighbor_tract)
            if neighbor_comm in candidate_comms:
                edge = self.enforcer.g_ig.es.find(_between=([tract_idx], [neighbor_idx]))
                edge_weights[neighbor_comm] += edge['weight'] if 'weight' in edge.attributes() else 1.0

        if not edge_weights:
            return list(candidate_comms)[0]

        return max(edge_weights.items(), key=lambda x: x[1])[0]

    def find_min_impact_score(self, tract, candidate_comms, score_metric, baseline_score):
        """Finds the community reassignment that minimises impact on quality metrics.

        Args:
            tract (str): The tract ID being reassigned.
            candidate_comms (set): Set of candidate community IDs to consider.
            score_metric (str): Quality metric to optimise ('modularity' or 'codelength').
            baseline_score (float): Original score before any reassignments.

        Returns:
            tuple: (best_community, resulting_score) pair

        Note:
            For modularity, higher scores are better. For codelength, lower scores are better.
            Always reverts the test assignment before returning the final decision.
        """
        original_comm = self.enforcer.community_dict[tract]
        best_comm = None

        if score_metric == 'modularity':
            best_delta = float('-inf')
        else:
            best_delta = float('inf')

        for comm in candidate_comms:
            self.enforcer._reassign_tract(tract, comm)

            if score_metric == 'modularity':
                membership = [
                    self.enforcer.community_dict.get(self.enforcer.graph_idx_to_tract[v.index], original_comm)
                    for v in self.enforcer.g_ig.vs
                ]
                current_score = self.enforcer.g_ig.modularity(membership, weights='weight')
                delta = current_score - baseline_score
                if delta > 0 and delta > best_delta:
                    best_delta = delta
                    best_comm = comm
                elif best_comm is None and abs(delta) < abs(best_delta):
                    best_delta = delta
                    best_comm = comm

            elif score_metric == 'codelength':
                im = Infomap("--two-level --silent")
                for edge in self.enforcer.g_ig.es:
                    im.add_link(edge.source, edge.target, edge["weight"])
                im.run(no_infomap=True)
                current_score = im.codelength
                delta = current_score - baseline_score
                if delta < 0 and delta < best_delta:
                    best_delta = delta
                    best_comm = comm
                elif best_comm is None and abs(delta) < abs(best_delta):
                    best_delta = delta
                    best_comm = comm

            self.enforcer._reassign_tract(tract, original_comm)

        if best_comm is not None:
            self.enforcer._reassign_tract(tract, best_comm)
            final_score = baseline_score + best_delta
        else:
            self.enforcer._reassign_tract(tract, original_comm)
            final_score = baseline_score

        return best_comm, final_score
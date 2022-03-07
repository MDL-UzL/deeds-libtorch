
import torch
from collections import OrderedDict
class Edge():
    def __init__(self, node_a, node_b, weight=0.):
        self.node_a = node_a
        self.node_b = node_b
        self.weight = weight

    def get_nodes(self):
        return (self.node_a, self.node_b)

    def __str__(self):
        return f"{self.node_a.coords} <> {self.node_b.coords}"

    def __repr__(self):
        return f"{self.node_a.coords} <> {self.node_b.coords}"
class PatchNode():
    def __init__(self, coords:list,
        _id=None, patch_content=None, parent=None, level=None):
        self.coords = coords
        self._id = _id
        self.patch_content = patch_content
        self.parent = parent
        self.level = level
        self.children_cost = 0.

    def __getitem__(self, idx):
        return self.coords[idx]

    def __str__(self):
        return str(self.coords)

    def __repr__(self):
        return str(self.coords)

def build_neighbourhood_adjacency(pd, pw, ph):
    node_count = pd*pd*pw
    six_neighbourhood = torch.tensor([[-1,  0,  0],
                                      [ 1,  0,  0],
                                      [ 0, -1,  0],
                                      [ 0,  1,  0],
                                      [ 0,  0, -1],
                                      [ 0,  0,  1]])
    nodes = OrderedDict()
    edges = []

    for d_idx in range(pd):
        for h_idx in range(ph):
            for w_idx in range(pw):
                c_list = [d_idx, h_idx, w_idx]

                nodes[tuple(c_list)] = PatchNode(
                    c_list,
                    _id=pd*ph*c_list[0] + ph*c_list[1] + c_list[2]
                )

    for d_idx in range(pd):
        for h_idx in range(ph):
            for w_idx in range(pw):
                c_list = [d_idx, h_idx, w_idx]
                current_coords = torch.tensor(c_list)
                current_node = nodes[tuple(c_list)]

                for neighbour_offset in six_neighbourhood:
                    neighbour_coords = current_coords + neighbour_offset

                    if any(neighbour_coords < 0):
                        continue
                    if any(neighbour_coords >= torch.tensor((pd,ph,pw))):
                        continue
                    if not any(neighbour_coords > current_coords):
                        continue

                    n_list = neighbour_coords.tolist()
                    neighbour_node = nodes[tuple(n_list)]
                    edges.append(Edge(current_node, neighbour_node))

    return nodes, edges

def get_patch_features(feature_volume, node_a, node_b, patch_len):
    vox_a_slices = [slice(patch_len*ord, patch_len*ord+patch_len) \
        for ord in node_a]
    vox_b_slices = [slice(patch_len*ord, patch_len*ord+patch_len) \
        for ord in node_b]
    return feature_volume[vox_a_slices], feature_volume[vox_b_slices]

def weight_from_edgecost(edgecost, feature_std):
    weight = (-edgecost/(2.0*feature_std)).exp()
    return weight

def calc_prims_graph(feature_volume, patch_len):

    pd, pw, ph = (torch.tensor(feature_volume.shape) / patch_len).int().tolist()
    nodes, edges = build_neighbourhood_adjacency(pd, pw, ph)

    # Prepare edge weights
    for edg in edges:
        node_a, node_b = edg.get_nodes()
        feat_a, feat_b = get_patch_features(feature_volume, node_a, node_b, patch_len)
        node_a.patch_content, node_b.patch_content = feat_a, feat_b
        # Calculate SAD of (mind) features of patches
        patch_difference = (feat_a - feat_b).abs().sum()
        print()
        edg.weight += patch_difference/feat_a.numel()

    feature_std = feature_volume.std(unbiased=False)

    for edg in edges:
        edg.weight = -weight_from_edgecost(edg.weight, feature_std)

    # Run prims algorithm
    edges.sort(key=lambda edg: edg.weight)

    root_node = nodes[(int(pd/2), int(pw/2), int(ph/2))]
    root_node.parent = root_node
    root_node.level = 0
    connected_node = root_node

    mst_edges = []
    mst_nodes = []
    mst_nodes.append(root_node)

    it_edges = iter(edges)
    while len(mst_edges) < len(nodes)-1:
        edge = next(it_edges)
        connected_node = set(mst_nodes).intersection(set(edge.get_nodes()))
        if connected_node:
            connected_node = list(connected_node)[0]
            unconnected_nb_node = set(edge.get_nodes()) - set(mst_nodes)
            if unconnected_nb_node:
                unconnected_nb_node = list(unconnected_nb_node)[0]
                unconnected_nb_node.parent = connected_node
                unconnected_nb_node.level = unconnected_nb_node.parent.level + 1
                unconnected_nb_node.parent.children_cost -= edge.weight
                mst_edges.append(edge)
                mst_nodes.append(unconnected_nb_node)
                del edges[edges.index(edge)]
                it_edges = iter(edges)

    return mst_nodes, mst_edges
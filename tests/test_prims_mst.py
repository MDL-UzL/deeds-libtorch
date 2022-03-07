import os
import unittest
from pathlib import Path
import torch
import timeit

from __init__ import CPP_DEEDS_MODULE, test_equal_tensors, log_wrapper
from deeds_libtorch.prims_mst import calc_prims_graph

class TestPrimsMst(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_calc_prims_graph(self):
        #########################################################
        # Prepare inputs
        GRID_DIVISOR = torch.tensor(1).int()
        D,H,W = 4,4,4
        mind_image = torch.arange(D*H*W).reshape(D,H,W).float()
        #########################################################
        # Get cpp output
        ordered, parents, edgemst = log_wrapper(CPP_DEEDS_MODULE.prims_mst_prims_graph, mind_image, GRID_DIVISOR)

        #########################################################
        # Get torch output
        patch_list, edge_list = log_wrapper(calc_prims_graph, mind_image, GRID_DIVISOR)
        patch_ids = [patch._id for patch in patch_list]
        parent_ids = [patch.parent._id for patch in patch_list]
        unary_children_costs = torch.tensor([patch.children_cost for patch in patch_list])

        #########################################################
        # Assert difference
        assert test_equal_tensors(edgemst.sum(), unary_children_costs.sum()), "Tensors do not match"


if __name__ == '__main__':
    # unittest.main()
    tests = TestPrimsMst()
    tests.test_calc_prims_graph()

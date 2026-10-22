"""Exercise the real pure grid builder without a GPU/JAX installation."""
import ast
from pathlib import Path
import unittest
import numpy as np


class FlowGridTests(unittest.TestCase):
    def grid(self,*args):
        path=Path(__file__).parents[1]/'inference/rtc/sampling.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='flow_time_grid')
        scope={'jnp':np}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
        return scope['flow_time_grid'](*args)

    def test_original_grids_unchanged(self):
        np.testing.assert_allclose(self.grid(20,0),np.linspace(1,0,21))
        np.testing.assert_allclose(self.grid(20,.001),np.r_[np.linspace(1,.001,20),0])

    def test_geometric_end_has_fixed_evaluation_budget_and_strict_time_order(self):
        for steps,refinement in [(20,4),(20,6),(20,10),(30,4)]:
            grid=self.grid(steps,.001,refinement)
            self.assertEqual(len(grid),steps+1)
            self.assertEqual(grid[0],1.);self.assertEqual(grid[-1],0.)
            self.assertTrue(np.all(np.diff(grid)<0))
            self.assertAlmostEqual(sum(-np.diff(grid)),1.)
            self.assertEqual(sum(grid[:-1]<=.0200001),refinement)

    def test_invalid_refinement_is_rejected(self):
        for args in [(20,.001,0),(20,.001,20),(20,.02,4),(1,.001,1)]:
            with self.assertRaises(ValueError):self.grid(*args)


if __name__=='__main__':unittest.main()

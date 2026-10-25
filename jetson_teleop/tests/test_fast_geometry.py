"""Numerical equivalence on both nominal arms; no hardware access."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'OptUSBProj-ubuntu'))
from jetson_teleop.benchmark_ik import model, tasks
from jetson_teleop.fast_geometry import enable


class FastGeometryTests(unittest.TestCase):
    def test_chain_and_linear_residuals_both_models(self):
        rng=np.random.default_rng(219)
        for side in ('right','left'):
            arm=model(side); enable(arm)
            qs=[arm.neutral,arm.reference,arm.neutral+1e-9,arm.neutral+1e-5,
                arm.lower,arm.upper,*rng.uniform(arm.lower,arm.upper,size=(100,6))]
            for q in qs:
                for positions in (None, np.zeros((2,3))):
                    directions=rng.normal(size=(2,3))
                    base,prefix=arm._compiled_geometry(q,arm.neutral,directions,positions)
                    np.testing.assert_allclose(prefix,arm.prefixes(q),atol=2e-12,rtol=2e-12)
                    s,e,w=arm.points(q)
                    expected=([5*(e-s-arm.lengths[0]*directions[0]),
                               5*(w-e-arm.lengths[1]*directions[1])] if positions is None else
                              [5*w,2.5*e])
                    expected.extend([.06*(q-arm.neutral),.008*(q-arm.neutral)])
                    np.testing.assert_allclose(base,np.concatenate(expected),atol=2e-12,rtol=2e-12)

    def test_fit_matches_reference_across_modes_and_palm_options(self):
        for side in ('right','left'):
            reference, fast=model(side),model(side);enable(fast)
            for task in tasks(reference,5):
                for mode in ('joint','staged'):
                    for palm in (True,False):
                        for positions in (True,False):
                            inputs=dict(task)
                            if not palm: inputs['rotation']=None
                            if not positions: inputs['positions']=None
                            inputs.update(previous=reference.neutral.copy(),step=np.full(6,2*np.pi),solver_mode=mode)
                            a=reference.fit(**inputs); b=fast.fit(**inputs)
                            np.testing.assert_allclose(a[0],b[0],atol=2e-5,rtol=0)
                            self.assertAlmostEqual(a[1],b[1],places=6)
                            if palm:self.assertAlmostEqual(a[2],b[2],places=3)


if __name__=='__main__': unittest.main()

"""Quick test suite: python -m unittest test_simple_energy_lab.py -v"""
import unittest
import torch
from simple_energy_lab import (LabConfig,SimpleSegmentationEnergy,sample_states,
                               simplex_project,tangent,state_gradient,descent,
                               finite_difference_direction,hessian_direction,
                               analyze_model,save_checkpoint,load_checkpoint)
from tempfile import TemporaryDirectory
from pathlib import Path

class TestEnergyLab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        cls.cfg=LabConfig(width=6,image_size=24,batch_size=1,curvature_directions=1,descent_steps=4)
        cls.model=SimpleSegmentationEnergy(6)
        cls.states=sample_states(cls.cfg)

    def test_simplex_projection(self):
        v=torch.randn(2,3,12,12)*4
        q=simplex_project(v)
        self.assertTrue((q>=0).all())
        self.assertTrue(torch.allclose(q.sum(1),torch.ones_like(q[:,0]),atol=1e-6))
        self.assertLess((simplex_project(q)-q).abs().max(),1e-6)

    def test_energy_and_gradient_shape(self):
        p=self.states['mild']
        self.assertEqual(tuple(self.model(p).shape),(1,))
        g=state_gradient(self.model,p)
        self.assertEqual(tuple(g.shape),tuple(p.shape))
        self.assertTrue(torch.isfinite(g).all())
        self.assertLess(tangent(g).sum(dim=1).abs().max().item(),1e-6)

    def test_directional_curvature_matches_hvp(self):
        p=self.states['interior']
        v=torch.randn_like(p)
        fd=finite_difference_direction(self.model,p,v,0.025)
        exact=hessian_direction(self.model,p,v)
        self.assertAlmostEqual(fd['finite_difference'],exact,delta=max(0.025*abs(exact),2e-3))

    def test_projected_descent_nonincreasing(self):
        p=self.states['strong']
        final,history=descent(self.model,p,steps=4,step_size=1.0)
        values=[s['energy_per_pixel'] for s in history]
        self.assertTrue(all(v <= u+1e-6 for u,v in zip(values,values[1:])))
        self.assertTrue(torch.allclose(final.sum(1),torch.ones_like(final[:,0]),atol=1e-6))

    def test_report_and_checkpoint(self):
        report=analyze_model(self.model,self.cfg)
        self.assertIn('curvature',report)
        self.assertIn('logit_limits',report)
        with TemporaryDirectory() as temp:
            path=Path(temp)/'demo.pt'
            save_checkpoint(self.model,path)
            loaded=load_checkpoint(path)
            self.assertTrue(torch.allclose(self.model(self.states['clean']),loaded(self.states['clean'])))

if __name__=='__main__':
    unittest.main()

"""Fast tests independent of psfmodels, checkpoints and painted templates."""
import numpy as np
import torch
from scipy import ndimage
from refinement_ablations import ABLATIONS, make_energy
from mm_fixed_compatible import prob_to_reduced
from geodesic_instances import WatershedConfig, geodesic_watershed, instance_score, evaluate_prediction
from run_refinement_study import StudyConfig, build_validation_data, run_study


def test_well_coefficient_and_preserved_symmetric_energy():
    torch.manual_seed(1)
    p = torch.rand(1,3,12,12); p=p/p.sum(1,keepdim=True)
    q=prob_to_reduced(p)
    model1 = make_energy(ABLATIONS[1])
    model2 = make_energy(ABLATIONS[2])
    w1 = model1.energy_terms_reduced(q)['triple_well']
    w2 = model2.energy_terms_reduced(q)['triple_well']
    u,w,z=p[:,1],p[:,2],p[:,0]
    expected = (model1.cfg.lam / model1.cfg.epsilon) * (
        2*u.square()*z.square()+u.square()*w.square()).sum()
    assert torch.allclose(w2-w1,expected[None],atol=1e-5)
    assert torch.allclose(model1.energy_terms_reduced(q)['dirichlet'],
                          model2.energy_terms_reduced(q)['dirichlet'])


def test_radius_limit_and_seed_identity():
    yy,xx=np.mgrid[:64,:64]
    seed=(yy-32)**2+(xx-32)**2 <= 1
    grown=geodesic_watershed(seed,np.zeros((64,64)), WatershedConfig(radius=5))
    assert grown.max()==1
    assert np.all(grown[ndimage.distance_transform_edt(~seed)>5]==0)
    assert np.all(grown[seed]==1)
    assert grown[32,37]==1
    seeds=np.zeros((64,64),bool); seeds[20,20]=True; seeds[20,26]=True
    labels=geodesic_watershed(seeds,np.zeros((64,64)),WatershedConfig(radius=8))
    assert labels[20,20]!=labels[20,26]
    assert labels[20,20]>0 and labels[20,26]>0


def test_instance_score():
    truth=np.zeros((20,20),int); truth[2:5,2:5]=1; truth[12:16,12:16]=2
    out=instance_score(truth,truth)
    assert (out['tp'],out['fp'],out['fn'],out['instance_f1'],out['pq'])==(2,0,0,1.,1.)
    out=instance_score(np.zeros_like(truth),truth)
    assert (out['tp'],out['fn'],out['instance_f1'])==(0,2,0.)


def _fixture(_):
    h=w=32; z=2
    imgs=np.zeros((z,2,h,w),dtype=np.float32)
    labels=np.zeros((z,3,h,w),dtype=np.float32)
    yy,xx=np.mgrid[:h,:w]
    for zi in range(z):
        nuc=(yy-12)**2+(xx-11-zi)**2<16
        labels[zi,0,nuc]=1
        labels[zi,1,((yy-12)**2+(xx-11-zi)**2<64)&~nuc]=1
        imgs[zi,1]=labels[zi,1]
    return imgs,labels


def test_end_to_end_fixture(tmp_path):
    cfg=StudyConfig(n_volumes=1,n_slices_per_volume=1,image_size=32,
                    z_slices=2,noise_levels=(0.8,),horizons=(0,1,2),
                    output_dir=str(tmp_path),save_previews=False,device='cpu')
    samples=build_validation_data(cfg,volume_loader=_fixture)
    rows,summary,traces=run_study(samples,cfg,selected=ABLATIONS[:3])
    assert len(rows)==3*3 and len(summary)==9 and len(traces)==9
    assert (tmp_path/'summary.csv').exists()
    for item in summary:
        assert np.isfinite(item['micro_instance_f1'])


def test_template_only_and_mixed_energy(tmp_path):
    from PIL import Image
    template = np.full((21,21,3), (63,72,204), dtype=np.uint8)
    template[8:13,8:13] = (0,0,0)
    template[5:8,8:13] = (255,255,255)
    Image.fromarray(template).save(tmp_path/'test_pattern.png')
    a=make_energy(ABLATIONS[3],template_dir=tmp_path, orientations=2)
    b=make_energy(ABLATIONS[5],template_dir=tmp_path, orientations=2)
    p=torch.softmax(torch.randn(1,3,32,32),dim=1)
    ta=a.energy_terms_reduced(prob_to_reduced(p))
    tb=b.energy_terms_reduced(prob_to_reduced(p))
    assert ta['templates'].shape==(1,) and tb['templates'].shape==(1,)
    assert ta['dirichlet'].item()==0 and ta['triple_well'].item()==0
    assert tb['dirichlet'].item()>0
    assert torch.allclose(ta['templates'],tb['templates'])


def test_ignore_ring_does_not_punch_holes():
    h = w = 31
    yy, xx = np.mgrid[:h, :w]
    prob = np.zeros((3, h, w), dtype=np.float32)
    seed = (yy - 15)**2 + (xx - 15)**2 <= 4
    prob[1, seed] = 1.0
    prob[0, ~seed] = 1.0
    target = np.zeros((h, w), dtype=np.int32)
    target[(yy - 15)**2 + (xx - 15)**2 <= 25] = 1
    ring = ((yy - 15)**2 + (xx - 15)**2 <= 36) & ((yy - 15)**2 + (xx - 15)**2 > 25)
    target[ring] = -100
    stats, maps = evaluate_prediction(prob, target, np.zeros((h, w), dtype=np.float32),
                                      WatershedConfig(radius=6), return_maps=True)
    assert maps['grown'][15, 15] > 0
    assert np.all(maps['grown'][seed] > 0)
    assert stats['pred_instances'] == 1
    assert np.all(maps['ground_truth'][ring] > 0)  # GT -100 cannot cut holes after expansion


def test_radius_diagnostic_recovers_match_at_zero_radius():
    from diagnose_instance_f1 import diagnose_prediction
    h=w=32
    yy,xx=np.mgrid[:h,:w]
    mask=(yy-16)**2+(xx-16)**2<=16
    target=np.zeros((h,w),dtype=int)
    target[mask]=1
    prob=np.zeros((3,h,w),dtype=np.float32)
    prob[0]=1
    prob[0,mask]=0
    prob[1,mask]=1
    report,details=diagnose_prediction(prob,target,radii=(0,5))
    assert report[0]['tp']==1 and report[0]['f1']==1.
    assert report[1]['f1']==1.  # equal seeds expanded by equal radii still match
    assert details[0]['best_iou']==1.
    assert details[1]['best_iou']==1.
    assert details[0]['best_iou']==1.


def test_same_expansion_for_prediction_and_ground_truth():
    h = w = 51
    yy, xx = np.mgrid[:h, :w]
    mask = ((yy - 25)**2 + (xx - 25)**2 <= 9)
    target = np.zeros((h, w), dtype=np.int32)
    target[mask] = 1
    # An ignored ring around the reference marker is no growth barrier.
    ignored_ring = ((yy - 25)**2 + (xx - 25)**2 <= 25) & ~mask
    target[ignored_ring] = -100
    prob = np.zeros((3, h, w), dtype=np.float32)
    prob[0] = 1.0
    prob[0, mask] = 0.0
    prob[1, mask] = 1.0
    config = WatershedConfig(radius=5)
    stats, maps = evaluate_prediction(prob, target, np.zeros((h, w)),
                                      config, return_maps=True)
    assert stats['instance_f1'] == 1.0
    assert stats['pq'] == 1.0
    assert np.array_equal(maps['ground_truth'], maps['grown'])
    assert np.any(maps['ground_truth'][ignored_ring] > 0)

def test_ground_truth_and_prediction_have_same_radius_cap():
    h = w = 45
    target = np.zeros((h, w), dtype=np.int32)
    target[15, 15] = 1
    prob = np.zeros((3, h, w), dtype=np.float32)
    prob[0] = 1
    prob[0, 25, 25] = 0
    prob[1, 25, 25] = 1
    stats, maps = evaluate_prediction(prob, target, np.zeros((h,w)),
                                      WatershedConfig(radius=5), return_maps=True)
    for seeds, instances in [(maps['ground_truth_seeds'], maps['ground_truth']),
                             (maps['seeds'], maps['grown'])]:
        assert np.all(instances[ndimage.distance_transform_edt(~seeds) > 5] == 0)
        assert instances[seeds].min() > 0


def test_centroid_one_to_one_and_distance():
    from geodesic_instances import centroid_score
    gt = np.zeros((30,30), np.int32)
    gt[5,5] = 1; gt[20,20] = 2
    pred = np.zeros_like(gt)
    pred[5,6] = 1; pred[5,7] = 2; pred[20,24] = 3
    result = centroid_score(pred, gt, distance_threshold=2)
    assert result['centroid_tp'] == 1
    assert result['centroid_fp'] == 2 and result['centroid_fn'] == 1
    assert np.isclose(result['centroid_f1'], 2/(2+2+1))
    assert centroid_score(pred, gt, distance_threshold=4)['centroid_tp'] == 2


def test_all_metrics_perfect_match_with_ignore_ring():
    from geodesic_instances import evaluate_prediction
    y = np.zeros((33,33), np.int32)
    y[10:15, 10:15] = 1
    y[9:16, 9] = -100
    p = np.zeros((3,33,33), np.float32)
    p[0] = 1
    p[0, 10:15, 10:15] = 0
    p[1, 10:15, 10:15] = 1
    score = evaluate_prediction(p,y,np.zeros_like(y),WatershedConfig(radius=5),centroid_distance_threshold=2)
    assert score['instance_f1'] == 1
    assert score['unexpanded_instance_f1'] == 1
    assert score['centroid_f1'] == 1
    assert np.isfinite(score['dice_macro'])


def test_final_table_and_three_class_images(tmp_path):
    from run_refinement_study import write_final_ablation_tables
    cfg=StudyConfig(n_volumes=3,n_slices_per_volume=1,image_size=32,
                    z_slices=2, noise_levels=(0.8,), horizons=(0,1),
                    example_volumes=3, centroid_distance_threshold=5.,
                    output_dir=str(tmp_path),save_previews=True,device='cpu')
    samples = build_validation_data(cfg, volume_loader=_fixture)
    rows, summary, traces = run_study(samples, cfg, selected=ABLATIONS[:2])
    assert len(rows) == 3*2*2
    assert all('centroid_f1' in row and 'unexpanded_instance_f1' in row for row in rows)
    assert all('dice_macro' in row for row in summary)
    final, file = write_final_ablation_tables(summary,cfg.horizons,tmp_path)
    assert len(final)==2 and file.exists() and (tmp_path/'final_ablation_table.csv').exists()
    figs=list(tmp_path.glob('three_class_*png'))
    assert len(figs)==2
    assert all(f.stat().st_size>1000 for f in figs)

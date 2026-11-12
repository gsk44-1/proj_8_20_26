# Controlled segmentation-energy refinement study

## Contents
- `Refinement_Ablations_Colab.ipynb`: setup, configuration, plots and saved outputs.
- `refinement_ablations.py`: asymmetric pairwise well + six ablation variants.
- `geodesic_instances.py`: radius-limited geodesic multi-seed watershed and instance matching.
- `run_refinement_study.py`: reproducible corrupted-GT validation + exported CSVs and PNGs.
- `test_refinement_study.py`: fast numerical and integration tests, not requiring the full synthetic generator.
- Original `mm_fixed_compatible.py`, `template_energy.py`, `anchored_refinement.py`, `synthetic_fluor_stains_three_classes_weighted.py`, and other dependency files copied without modification.

## Files you must supply for all six ablations
The eight **original** 21x21 painted template PNGs from your earlier project are *not* in the current uploads. Put them in the `templates/` folder, preserving the names and exact PNG colors. The experiment runner does not invent substitutes. MM-only tests run without these templates. If you want to evaluate previously learned strengths, set `template_checkpoint` to the saved `best.pt` and provide the matching original templates. Otherwise strengths are fixed to 0.05.

## In Colab
1. Upload `segmentation_refinement_tests.zip` (this package) and unzip it under `/content/segmentation_refinement_tests` or mount Drive. Open the notebook, update `PROJECT_DIR` accordingly.
2. The notebook installs missing dependencies including `psfmodels`, then runs a unit-test check.
3. Upload or copy your eight template PNGs into `PROJECT_DIR/templates/` for all six ablations.
4. Start with a small pilot (e.g., 1 volume, 2 slices, 0,10,25 steps); then enlarge `StudyConfig` for meaningful validation and use a held-out volume seed for reporting.

## Evaluation specification
- Input is **perturbed ground truth**, generated once per slice per noise strength with the preexisting `synthetic_logits` spectral corruption and a fixed blur sigma; all models see identical probability fields.
- 2D nuclear seeds = 8-connected components of predicted class-1 argmax, excluding ignored GT-overlap pixels.
- Original **nonnuclear fluorescence image**, separately normalized at percentiles 1 and 99, sets a positive geodesic traversal cost `1 + alpha * normalized_intensity` (alpha=4 by default).
- Competitive 8-neighbor Dijkstra propagation uses the *mean* endpoint cost times Euclidean neighbor length. No unlabeled pixel is forcibly filled; expansion can never extend farther than **25 Euclidean pixels from the original connected-component seed**. Per-seed labels stay distinct when they meet.
- Ground truth = 8-connected components of class-1 GT binary mask; no expansion. Ignored overlapping stain labels are excluded from reference and predicted evaluation.
- Matching is one-to-one maximum-cardinality IoU>=0.5; pooled F1 aggregates TP/FP/FN across slices. Files include PQ, pre-/post-growing Dice, splits and merges, energy components, previews, and F1-vs-step plots.
- The no-refinement control returns the original predictions unchanged. All other variants use the same fixed step size and L2 anchor. "Templates only" disables **both** MM Dirichlet and well terms (`lam=0`) while retaining the anchor.

## Scope and limitations
- This is **validation/evaluation**, not a new trainer. It supports fixed template strengths or loading saved strengths, but it does not retrain coefficients for the asymmetric potential. Using an old checkpoint with a new potential is an evaluation transfer and should be labeled as such.
- Slices from the same volume are correlated; for presentation-quality uncertainty intervals, bootstrap by **volume**, and compare held-out volumes.
- The initial state is generated *from the ground truth*, so performance on corrupted-GT inputs does not establish generalization to U-Net predictions or real images.
- The planted nucleus IDs from the synthetic generator are not returned by the dataset interface; the GT instance definition here follows the agreed 8-connected nuclear binary mask.
- The old energy implementation uses periodic (circular) image boundaries, unchanged here.

## Running tests locally
```bash
pip install torch scipy numpy pillow matplotlib scikit-image numba psfmodels pytest
cd segmentation_refinement_tests
python -m pytest -q test_refinement_study.py
```

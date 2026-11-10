# Experiment B — Rotated learned filter bank (Colab)

Changes from the previous Experiment B:

1. New extra feature strengths initialize to **0.05** (previously 0.002), with cap 0.3.
2. Each of 2–4 learned two-channel canonical filters is rotated to **16 orientations across 360 degrees** by default. Rotations use the `_scipy_rotation_matrices` helper in `mm_fixed_compatible.py`, preserving SciPy bilinear `rotate` conventions and differentiability. Each rotated kernel is zero-centered per channel and normalized to positive coefficient sum 1. Orientation responses are max-pooled per feature, then ReLU and squared in the negative reward energy.
3. Existing separator and boundary shapes, MM energy, shared parameters over time, anchor, and supervised CE + Dice remain unchanged.
4. Notebook defaults to SGD learning rate **1.0**, momentum 0, as in the previous successful test; adjust as desired.

## Files you need to replace

For an existing Colab Experiment B setup, replace **`learned_energy.py`** and **`train_experiment_b.py`**. Replace the visualization cell (if used) with **`experiment_b_visualization_cell.py`** to load the new orientation config from checkpoints. The rest of your original Experiment B Python files are included unchanged for convenience.

To run standalone, upload `experiment_b_rotated_bundle.zip` to the notebook; its setup cell extracts to `/content/experiment_b_rotated/`.

## New / changed configuration

```python
n_extra=3,
extra_filter_size=13,
extra_orientations=16,
initial_extra_strength=0.05,
learning_rate=1.0,
output_dir='/content/drive/MyDrive/Colab Notebooks/proj_26/experiment_b_rotated_results',
```

## Checkpoints

Use a **fresh output directory**. Older Experiment B state_dict files were built using non-rotated learned features and do not contain the `extra_rotmat` buffer; they cannot be loaded strictly into the new structure. The frozen U-Net checkpoint remains unchanged.

## Practical considerations

The 3×16 filter bank adds 48 oriented convolutions per refinement step, so 400-step evaluation will be slower. The underlying canonical parameter count remains 1019 for 3 filters of size 13, because rotated copies share trainable coefficients. Orientation invariance is approximate (finite angle sampling and discrete-grid interpolation). No end-to-end training was executed on the actual Colab microscopy volumes or checkpoint; a local synthetic gradient and normalization smoke test was run.

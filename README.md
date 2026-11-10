# Experiment B: shared structured energy for nuclear refinement

The frozen pretrained U-Net supplies three-class softmax probabilities. The energy is the **original Modica–Mortola term and fixed separator/boundary filters**, plus 2–4 new learned zero-DC, positive-sum-normalized convolution filters. Each family has a bounded positive strength. Anchor fidelity to the *original* probability map remains fixed, \(\mu=0.4\), with full-three-class L2 fidelity and the original reduced-coordinate metric.

Default strengths: separator 0.5, boundary 0.5 (caps 1.5), new filters 0.002 (caps 0.3). Shared parameters across all PGD steps. Fixed step size 0.02, no backtracking in differentiable training. Default optimizer **SGD** lr=0.001, momentum=0, weight decay=1e-5; these match the requested learning rate and weight decay but may require LR tuning due to tiny higher-order gradients. Four differentiable steps follow detached randomized warmup (0,20,80,160,300). Optional blur and spectral restart corruption is applied to the evolved state. Training input is chosen per update from noisy/blurred ground truth or frozen U-Net probabilities. Both branches anchor their PGD trajectories to the selected initial input.

Training loss reproduces the supplied U-Net objective: class-weighted CE with spatial weights \(1+2m\), CE weights (1,2,1), class-weighted soft Dice weights (0.2,0.4,0.4), Dice multiplier 1.5. Small parameter regularizers are added.

At evaluation time, 8-connected nuclear components produce instance F1@IoU0.5, merge and split rates; also semantic class Dice. Validation examples are selected from multiple held-out volumes. Best checkpoint is selected by held-out long-horizon instance F1. This validation metric is discrete and **is not differentiated**.

Files: `Experiment_B_Colab.ipynb`, `train_experiment_b.py`, `learned_energy.py`, `experiment_b_loss.py` plus original generator, U-Net, energy modules and shared utilities.

**Cautions:** The dataset is generated lazily and cached in CPU RAM (train_volumes * 16 slices). Set train/val seeds to disjoint ranges and avoid any seeds used for U-Net training. `SGD(lr=1e-3)` may make imperceptibly small updates because the new filter strengths and their gradients start small; inspect `grad_norm` and learned strengths and consider larger LR, momentum, or Adam only after checking them. Long warmups and 400-step validation are compute-intensive. This is a starting experiment, not a tuned benchmark. Checkpoint and dataset are not bundled.

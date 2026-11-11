# Template energy experiment

1. Upload the complete ZIP to Colab and run `Template_Energy_Colab.ipynb`.
2. Eight exact 21x21 input PNGs live in `templates/`, next to `template_energy.py`. The module discovers them using `Path(__file__).parent / 'templates'` by default, or the explicit `template_dir` config.
3. The only learned weights are 8 capped strengths. Their initial values are 0.05 and their caps are 0.3. Existing MM, spectral corruption, U-Net inference, unrolling, SGD, weighted CE + Dice and instance evaluation reuse the previous components.
4. The original separator and boundary filter terms are disabled. The parent MM computation is preserved exactly. The parent class may still construct the old fixed filter buffers internally, but **their energy contributions are zero**.
5. Black -> (+1,-1), white -> (-1,+1), indigo -> (0,0), nuclear/nonnuclear channels respectively. Indigo is neutral in the unrotated filter. Rotational interpolation blends values locally; no global zero-mean subtraction is applied.
6. Use a new output directory: learned B checkpoints are not compatible.
7. The signed-contrast convention does not explicitly ask for background at any pixel. If you want background-sensitive painted templates in the future, introduce a fourth color and a three-channel template or derive an appropriate two-channel contrast with constant offset.

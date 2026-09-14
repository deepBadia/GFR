"""Shared helpers used by every per-dataset experiment folder under
``IA_mesure/experiments/``.

Each dataset folder (``absorbant/``, ``mesure_couplage/``, ``mesure_diag/``,
``mesure_gain/``, ``secteur1/``, ``secteur1_amp_phase/``) only contains a
``config.yaml`` plus a handful of thin scripts (train baseline, train with
active learning, tune hyperparameters, compare models). All the logic that
is identical across datasets lives here so it is written -- and fixed -- once.
"""

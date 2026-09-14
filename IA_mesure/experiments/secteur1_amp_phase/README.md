# secteur1_amp_phase

Pipeline d'entrainement / active learning / tuning / comparaison pour `secteur1_amp_phase.h5`.

Voir [`../README.md`](../README.md) pour le fonctionnement general (boucle
d'active learning, correctifs apportes a `src/surmod`, structure des
sorties). Ce fichier ne liste que ce qui est specifique a ce dataset.

## Notes specifiques

Meme grille que secteur1.h5 (59 x 9) mais la sortie est deja stockee en (amplitude_linear, phase_degrees) plutot qu'en (S_real, S_imag). `complex_format: polar` reste desactive ici -- cette conversion suppose une entree (reel, imaginaire), pas une amplitude/phase deja calculee.

## Utilisation

```bash
cd IA_mesure/experiments/secteur1_amp_phase
python 00_visualize_data.py                # affiche le split train/val/test
python 01_train_gfr.py                     # baseline GFR-Net (tout le pool)
python 02_train_active_gfr.py              # Active-GFR-Net + boucle d'active learning
python 03_tune_hyperparams.py              # recherche d'hyperparametres Optuna
python 04_compare_models.py                # tableau + graphes de comparaison
```

Chaque script accepte des options pour des essais rapides sans toucher
`config.yaml`, par exemple :

```bash
python 01_train_gfr.py --epochs 20
python 02_train_active_gfr.py --n-rounds 3 --epochs-per-round 50
python 03_tune_hyperparams.py --n-trials 5 --epochs 100
```

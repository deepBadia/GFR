# mesure_diag

Pipeline d'entrainement / active learning / tuning / comparaison pour `mesure_diag.h5`.

Voir [`../README.md`](../README.md) pour le fonctionnement general (boucle
d'active learning, correctifs apportes a `src/surmod`, structure des
sorties). Ce fichier ne liste que ce qui est specifique a ce dataset.

## Notes specifiques

Diagramme de rayonnement : 52 geometries (theta_tgt, x, c, cut, freq, pol) x 451 points d'angle d'observation (theta_obs, -90..90 deg). `freq` n'a que 3 valeurs dans ce fichier, donc traite comme parametre discret plutot que comme axe continu.

## Utilisation

```bash
cd IA_mesure/experiments/mesure_diag
python 00_visualize_data.py                # affiche le split train/val/test
python 01_train_gfr.py                     # baseline GFR-Net (tout le pool)
python 02_train_active_gfr.py              # Active-GFR-Net + boucle d'active learning
python 03_tune_hyperparams.py              # recherche d'hyperparametres Optuna
python 04_compare_models.py                # tableau + graphes de comparaison
python 05_visualize_tuning.py              # graphes de la recherche Optuna
```

`01`, `02` et `03` acceptent `--cpus N` pour brider les threads PyTorch
(utile sur un cluster partage) -- voir `../README.md`.

Chaque script accepte des options pour des essais rapides sans toucher
`config.yaml`, par exemple :

```bash
python 01_train_gfr.py --epochs 20
python 02_train_active_gfr.py --n-rounds 3 --epochs-per-round 50
python 03_tune_hyperparams.py --n-trials 5 --epochs 100
```

# mesure_gain

Pipeline d'entrainement / active learning / tuning / comparaison pour `mesure_gain.h5`.

Voir [`../README.md`](../README.md) pour le fonctionnement general (boucle
d'active learning, correctifs apportes a `src/surmod`, structure des
sorties). Ce fichier ne liste que ce qui est specifique a ce dataset.

## Notes specifiques

91 geometries (angle, colonne h5 nommee "freq") x 37 points de frequence (colonne h5 nommee "theta"). CONFIRME par le proprietaire des donnees : les noms de colonnes "theta" et "freq" sont inverses dans le fichier source (meme situation que `mesure_couplage.h5`) -- la colonne "theta" contient en realite la frequence (2-11 GHz, correspond exactement au balayage de `mesure_couplage.h5`), et la colonne "freq" contient en realite l'angle d'observation (-90..90 deg). config.yaml utilise deja le bon mapping (axis_column: theta, X_columns: [freq]).

## Utilisation

```bash
cd IA_mesure/experiments/mesure_gain
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

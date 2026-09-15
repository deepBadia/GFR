# mesure_couplage

Pipeline d'entrainement / active learning / tuning / comparaison pour `mesure_couplage.h5`.

Voir [`../README.md`](../README.md) pour le fonctionnement general (boucle
d'active learning, correctifs apportes a `src/surmod`, structure des
sorties). Ce fichier ne liste que ce qui est specifique a ce dataset.

## Notes specifiques

ATTENTION : les noms de colonnes du fichier source sont trompeurs. Confirme avec le proprietaire des donnees : `ant_nb` est en realite la frequence (2-11 GHz, 37 pts), `freq` est en realite le numero de couplage (1-8). config.yaml utilise deja le bon mapping (axis_column: ant_nb, X_columns: [freq, pol]), et `dataset.display_names` corrige aussi les libelles affiches sur les graphes (`00_visualize_data.py`, `06_plot_raw_data.py`) pour qu'ils montrent "freq (GHz)"/"coupling_id" plutot que les noms de colonnes h5 bruts. Seulement 16 geometries distinctes -> resultats d'active learning a prendre comme illustratifs uniquement.

## Utilisation

```bash
cd IA_mesure/experiments/mesure_couplage
python 00_visualize_data.py                # affiche le split train/val/test
python 01_train_gfr.py                     # baseline GFR-Net (tout le pool)
python 02_train_active_gfr.py              # Active-GFR-Net + boucle d'active learning
python 03_tune_hyperparams.py              # recherche d'hyperparametres Optuna
python 04_compare_models.py                # tableau + graphes de comparaison
python 05_visualize_tuning.py              # graphes de la recherche Optuna
python 06_plot_raw_data.py                  # coupes + carte des donnees brutes
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

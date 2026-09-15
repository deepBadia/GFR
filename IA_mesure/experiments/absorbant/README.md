# absorbant

Pipeline d'entrainement / active learning / tuning / comparaison pour `absorbant.h5`.

Voir [`../README.md`](../README.md) pour le fonctionnement general (boucle
d'active learning, correctifs apportes a `src/surmod`, structure des
sorties). Ce fichier ne liste que ce qui est specifique a ce dataset.

## Notes specifiques

Le plus gros dataset (124k lignes, 4000 geometries x 31 points de frequence). `n_geom_cont: 6` dans config.yaml separe les 6 parametres continus (d1..d6) des 3 parametres discrets (polarization, incidence_angle, s_type). C'est le meilleur candidat pour observer un vrai gain de l'active learning (4000 geometries candidates dans le pool).

**Seul dataset avec du bruit de mesure simule** : `common.measurement_noise_std: 0.02` dans config.yaml ajoute un bruit gaussien sur S_real/S_imag avant toute conversion polaire (voir `../README.md` et `src/surmod/core/data_loader.py`), pour se rapprocher d'une vraie mesure plutot que d'une simulation propre. Mettre a 0.0 pour revenir aux donnees propres.

## Utilisation

```bash
cd IA_mesure/experiments/absorbant
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

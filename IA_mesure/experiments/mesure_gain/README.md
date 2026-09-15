# mesure_gain

Pipeline d'entrainement / active learning / tuning / comparaison pour `mesure_gain.h5`.

Voir [`../README.md`](../README.md) pour le fonctionnement general (boucle
d'active learning, correctifs apportes a `src/surmod`, structure des
sorties). Ce fichier ne liste que ce qui est specifique a ce dataset.

## Notes specifiques

37 geometries (theta) x 91 points de frequence (freq). Voir le commentaire en tete de config.yaml : les plages de valeurs sont surprenantes pour ces noms de colonnes (freq va de -90 a 90, theta de 2 a 11) -- a revalider avec la source de mesure si les resultats semblent incoherents ; si besoin, inversez juste `axis_column` et `X_columns` dans config.yaml, rien d'autre ne change.

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

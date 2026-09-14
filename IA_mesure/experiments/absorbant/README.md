# absorbant

Pipeline d'entrainement / active learning / tuning / comparaison pour `absorbant.h5`.

Voir [`../README.md`](../README.md) pour le fonctionnement general (boucle
d'active learning, correctifs apportes a `src/surmod`, structure des
sorties). Ce fichier ne liste que ce qui est specifique a ce dataset.

## Notes specifiques

Le plus gros dataset (124k lignes, 4000 geometries x 31 points de frequence). `n_geom_cont: 6` dans config.yaml separe les 6 parametres continus (d1..d6) des 3 parametres discrets (polarization, incidence_angle, s_type). C'est le meilleur candidat pour observer un vrai gain de l'active learning (4000 geometries candidates dans le pool).

## Utilisation

```bash
cd IA_mesure/experiments/absorbant
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

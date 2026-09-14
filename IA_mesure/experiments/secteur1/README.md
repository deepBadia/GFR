# secteur1

Pipeline d'entrainement / active learning / tuning / comparaison pour `secteur1.h5`.

Voir [`../README.md`](../README.md) pour le fonctionnement general (boucle
d'active learning, correctifs apportes a `src/surmod`, structure des
sorties). Ce fichier ne liste que ce qui est specifique a ce dataset.

## Notes specifiques

59 geometries (x) x 9 points de frequence (y), sortie complexe (S_real, S_imag) convertie en polaire (gain_dB, sin(phase), cos(phase)). Memes donnees que IA_mesure/configsSecteur1.yaml, reprises ici pour que ce dossier soit autonome.

## Utilisation

```bash
cd IA_mesure/experiments/secteur1
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

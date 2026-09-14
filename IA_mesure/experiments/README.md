# Pipelines d'entraînement / active learning / tuning / comparaison

Un dossier par fichier de `GFR_NET/data/*.h5`, chacun avec les mêmes 4 scripts :

| Script | Rôle |
|---|---|
| `01_train_gfr.py` | Entraîne un `GFR_Net` classique sur tout le pool disponible (baseline). |
| `02_train_active_gfr.py` | Entraîne un `Active-GFR-Net` avec une boucle d'active learning (voir plus bas). |
| `03_tune_hyperparams.py` | Recherche d'hyperparamètres Optuna (`src/surmod/core/tuner.py`) pour le `GFR_Net` baseline. |
| `04_compare_models.py` | Évalue les checkpoints trouvés sur le **même** jeu de test, produit un tableau + des graphes. |

Chaque dossier a son propre `config.yaml` (système de templates déjà utilisé par `IA_mesure/configsSecteur1.yaml`), donc totalement autonome : `cd IA_mesure/experiments/<dataset>/ && python 01_train_gfr.py`.

Toute la logique commune (boucle d'active learning, métriques, rapport de comparaison) est factorisée dans `_common/` pour éviter de la dupliquer x6 et de la faire diverger — les scripts de chaque dossier ne font qu'appeler cette bibliothèque avec les bons noms d'expérience.

## Correspondance dataset -> config

| Dossier | Fichier source | Géométrie (X) | Axe continu | Sortie (Y) | Notes |
|---|---|---|---|---|---|
| `secteur1/` | `secteur1.h5` | `x` (59 valeurs) | `y` (9 pts) | `S_real, S_imag` (polaire) | Identique à `IA_mesure/configsSecteur1.yaml` |
| `secteur1_amp_phase/` | `secteur1_amp_phase.h5` | `x` (59 valeurs) | `y` (9 pts) | `amplitude_linear, phase_degrees` | **Pas** de `complex_format: polar` ici : la conversion suppose du (réel, imag), pas de l'amplitude/phase déjà calculée |
| `absorbant/` | `absorbant.h5` | `d1..d6, polarization, incidence_angle, s_type` (4000 combinaisons) | `freq_ghz` (31 pts) | `S_real, S_imag` (polaire) | `n_geom_cont: 6` -> d1..d6 passent par l'encodeur Fourier géométrique, les 3 paramètres discrets par une branche séparée. De loin le plus gros dataset (124k lignes) : le meilleur candidat pour démontrer un vrai gain de l'active learning |
| `mesure_diag/` | `mesure_diag.h5` | `theta_tgt, x, c, cut, freq, pol` (52 combinaisons) | `theta_obs` (451 pts, -90°..90°) | `gain` | Diagramme de rayonnement ; `freq` n'a que 3 valeurs -> traité comme paramètre discret plutôt que comme axe |
| `mesure_couplage/` | `mesure_couplage.h5` | `freq` = numéro de couplage, `pol` (16 combinaisons) | `ant_nb` = **vraie fréquence** 2-11 GHz (37 pts) | `value` (couplage dB) | ⚠️ Noms de colonnes trompeurs dans le fichier source (confirmé) : voir le commentaire en tête de `config.yaml`. Seulement 16 géométries -> résultats d'active learning à prendre comme illustratifs |
| `mesure_gain/` | `mesure_gain.h5` | `theta` (37 valeurs) | `freq` (91 pts) | `gain` | Voir la note dans `config.yaml` : les plages de valeurs de `freq`/`theta` sont surprenantes pour ces noms -- à revérifier de votre côté si les résultats semblent bizarres |

## Ce que fait la boucle d'active learning (`_common/active_learning.py`)

1. Départ avec un petit ensemble de géométries labellisées, choisies aléatoirement dans le pool (le jeu de test, lui, n'est **jamais** touché).
2. Entraînement d'un `Active-GFR-Net` sur ce sous-ensemble.
3. Évaluation sur le jeu de test (identique pour tous les runs -> comparaison honnête).
4. Score d'incertitude (MC-dropout, `ActiveGFRNet.uncertainty_score`) sur les géométries restantes du pool.
5. Ajout des géométries les plus incertaines à l'ensemble labellisé, retour à l'étape 2.

Le fichier `history.csv` produit (colonnes `n_labeled_geoms`, `weighted_mse`, ...) répond directement à l'objectif "réduire le nombre de mesures/simulations nécessaires" : `04_compare_models.py` superpose cette courbe à la performance du baseline entraîné sur 100% du pool (`active_learning_efficiency.png`).

**Important (correctif apporté à `src/surmod`) :** avant cette livraison, `core/trainer.train_model` ignorait silencieusement le sous-ensemble labellisé (`labeled_mask`) et entraînait toujours sur tout le pool, même en mode "active learning" -- le mécanisme existait dans `DataLoader`/`ActiveGFRNet` mais n'était jamais réellement branché. Un paramètre `new_points` a été ajouté à `core.trainer.train_model` (et exposé dans `surmod.main.train_model`) pour que ça fonctionne réellement de bout en bout. Deux autres petits bugs corrigés au passage (voir les commentaires dans le code) :
- un timestamp par défaut figé au premier import du module (`core/trainer.py`, `core/tuner.py`), qui faisait que plusieurs runs dans le même process s'écrasaient les uns les autres ;
- `raise optuna.TrialPruned()` sans jamais importer `optuna` dans `core/trainer.py` (plantait dès qu'Optuna élaguait un essai) ;
- un dossier `_tuning/trials/trial_None` créé pour n'importe quel entraînement `active_gfr_net`, même hors tuning.

## Lancer les choses en pratique

```bash
cd IA_mesure/experiments/secteur1
python 01_train_gfr.py                       # baseline complet (config.yaml)
python 02_train_active_gfr.py                # active learning
python 03_tune_hyperparams.py --n-trials 30 --time-hours 4   # tuning Optuna
python 04_compare_models.py                  # tableau + graphes de comparaison
```

Chaque script accepte des options en ligne de commande pour des tests rapides sans toucher `config.yaml` (`--epochs`, `--n-rounds`, `--n-trials`, ...) -- voir `--help` ou l'en-tête de chaque fichier.

Sorties (non versionnées, voir `.gitignore`) :
```
<dataset>/results/<data_stem>/
  baseline_GFR_Net_<timestamp>/                 # 01_train_gfr.py
  active_active_gfr_net_active_r00/ ...         # 02_train_active_gfr.py (un dossier par round)
  active_active_learning/history.csv            # historique "précision vs nb labellisé"
  baseline_GFR_Net_<timestamp>_tuning/           # 03_tune_hyperparams.py (meilleur essai)
<dataset>/comparison/
  comparison_summary.csv, comparison_loss.png, comparison_rmse.png,
  active_learning_efficiency.png
```

## Limites connues / à garder en tête

- **`mesure_couplage.h5`** n'a que 16 géométries distinctes : l'active learning n'a de sens qu'à très petite échelle ici (pas un cas d'usage représentatif pour juger de l'efficacité de la méthode).
- Les hyperparamètres par défaut dans chaque `config.yaml` sont des points de départ raisonnables, pas le résultat d'une recherche -- utilisez `03_tune_hyperparams.py` pour les affiner sérieusement.
- Le tuning Optuna (`03_tune_hyperparams.py`) ne concerne que le `GFR_Net` baseline (le tuner ne gère pas encore l'active learning, cohérent avec le commentaire déjà présent dans `core/tuner.py`: "supervised only for now").
- Testé sur ce sandbox en CPU uniquement (pas de GPU) avec des runs très courts (quelques epochs / rounds / essais) juste pour valider que le pipeline tourne sans erreur de bout en bout sur les 6 datasets -- les valeurs numériques observées ne sont pas représentatives de la performance finale. Relancez avec les budgets par défaut (voir chaque `config.yaml`) pour de vrais résultats.

# Pipelines d'entraînement / active learning / tuning / comparaison

Un dossier par fichier de `GFR_NET/data/*.h5`, chacun avec les mêmes 6 scripts :

| Script | Rôle |
|---|---|
| `00_visualize_data.py` | Affiche les géométries et le découpage train/validation/test (voir plus bas). |
| `01_train_gfr.py` | Entraîne un `GFR_Net` classique sur tout le pool disponible (baseline). |
| `02_train_active_gfr.py` | Entraîne un `Active-GFR-Net` avec une boucle d'active learning (voir plus bas). |
| `03_tune_hyperparams.py` | Recherche d'hyperparamètres Optuna (`src/surmod/core/tuner.py`) pour le `GFR_Net` baseline -- voir "Que fait `03_tune_hyperparams.py` exactement ?" plus bas. |
| `04_compare_models.py` | Évalue les checkpoints trouvés sur le **même** jeu de test, produit un tableau + des graphes. |
| `05_visualize_tuning.py` | Graphes de la recherche Optuna (historique, importances, pruning...) -- voir plus bas. |
| `06_plot_raw_data.py` | Trace les données brutes elles-mêmes (pas le split, pas des prédictions) : coupes et carte 2D -- voir plus bas. |

`01`, `02` et `03` acceptent tous `--cpus N` pour brider le nombre de threads CPU utilisés (utile sur un cluster partagé -- voir plus bas).

Chaque dossier a son propre `config.yaml` (système de templates déjà utilisé par `IA_mesure/configsSecteur1.yaml`), donc totalement autonome : `cd IA_mesure/experiments/<dataset>/ && python 01_train_gfr.py`.

Toute la logique commune (boucle d'active learning, métriques, rapport de comparaison) est factorisée dans `_common/` pour éviter de la dupliquer x6 et de la faire diverger — les scripts de chaque dossier ne font qu'appeler cette bibliothèque avec les bons noms d'expérience.

## Correspondance dataset -> config

| Dossier | Fichier source | Géométrie (X) | Axe continu | Sortie (Y) | Notes |
|---|---|---|---|---|---|
| `secteur1/` | `secteur1.h5` | `x` (59 valeurs) | `y` (9 pts) | `S_real, S_imag` (polaire) | Identique à `IA_mesure/configsSecteur1.yaml` |
| `secteur1_amp_phase/` | `secteur1_amp_phase.h5` | `x` (59 valeurs) | `y` (9 pts) | `amplitude_linear, phase_degrees` | **Pas** de `complex_format: polar` ici : la conversion suppose du (réel, imag), pas de l'amplitude/phase déjà calculée |
| `absorbant/` | `absorbant.h5` | `d1..d6, polarization, incidence_angle, s_type` (4000 combinaisons) | `freq_ghz` (31 pts) | `S_real, S_imag` (polaire) | `n_geom_cont: 6` -> d1..d6 passent par l'encodeur Fourier géométrique, les 3 paramètres discrets par une branche séparée. De loin le plus gros dataset (124k lignes) : le meilleur candidat pour démontrer un vrai gain de l'active learning. **Seul dataset avec du bruit de mesure simulé** (voir plus bas) |
| `mesure_diag/` | `mesure_diag.h5` | `theta_tgt, x, c, cut, freq, pol` (52 combinaisons) | `theta_obs` (451 pts, -90°..90°) | `gain` | Diagramme de rayonnement ; `freq` n'a que 3 valeurs -> traité comme paramètre discret plutôt que comme axe |
| `mesure_couplage/` | `mesure_couplage.h5` | `freq` = numéro de couplage, `pol` (16 combinaisons) | `ant_nb` = **vraie fréquence** 2-11 GHz (37 pts) | `value` (couplage dB) | ⚠️ Noms de colonnes trompeurs dans le fichier source (confirmé) : voir le commentaire en tête de `config.yaml`. Seulement 16 géométries -> résultats d'active learning à prendre comme illustratifs |
| `mesure_gain/` | `mesure_gain.h5` | `freq` = **vrai angle theta** -90°..90° (91 valeurs) | `theta` = **vraie fréquence** 2-11 GHz (37 pts) | `gain` | ⚠️ Noms de colonnes inversés dans le fichier source (confirmé, comme `mesure_couplage.h5`) : voir le commentaire en tête de `config.yaml` |

## Visualiser les données et le split (`00_visualize_data.py`, `_common/data_viz.py`)

Avant d'entraîner quoi que ce soit, `00_visualize_data.py` construit le `DataLoader` (pas de modèle, donc quasi instantané même sur `absorbant.h5`) et sauvegarde `comparison/data_split_overview.png` avec deux panneaux :

- **Géométries** : nuage de points dans l'espace des paramètres géométriques, coloré par train/val/test.
  - 1 paramètre -> tracé directement sur une ligne ;
  - 2 paramètres -> tracé directement en 2D ;
  - 3+ paramètres (ex. `absorbant/` avec 9) -> projection PCA en 2D (calculée à la main via SVD numpy, pas de dépendance sklearn).
- **Grille de couverture** : matrice (géométrie x point d'axe) colorée par split -- répond littéralement à "quels points sont sélectionnés pour l'entraînement et la validation". Utile aussi pour vérifier que le test set n'est pas juste "les derniers X% des lignes" : le split (`DataLoader._setup_initial_split`) utilise `np.linspace` sur les indices, donc train/val/test sont entrelacés sur tout l'espace des géométries, ce qui se voit bien sous forme de bandes verticales alternées sur la grille.

Couleurs (`_common/data_viz.py::SPLIT_COLORS`) : **train = bleu, val = vert, test = rouge** -- modifiable directement dans ce dict si besoin.

Le split affiché est celui de `01_train_gfr.py` (pool complet -> train/val, test toujours à part) ; il ne reflète pas l'état évolutif de l'active learning (qui interroge un sous-ensemble croissant du pool à chaque round).

## Visualiser les données brutes (`06_plot_raw_data.py`, `_common/raw_data_viz.py`)

Contrairement à `00_visualize_data.py` (le split) et à `generate_results` (les prédictions du modèle), `06_plot_raw_data.py` trace uniquement les **données mesurées/simulées elles-mêmes**, dénormalisées en unités physiques. Deux vues, sauvegardées dans `comparison/` :

- `raw_data_slices.png` (**coupes**) : `--n-samples` géométries choisies régulièrement dans le pool (6 par défaut), sortie tracée en fonction de l'axe continu, une courbe par géométrie, un panneau par canal de sortie (ex. `gain_dB`/`sin_phase`/`cos_phase` pour les datasets polaires). Légende commune sous la figure (pas une par panneau, sinon ça déborde sur le panneau voisin avec des libellés de géométrie longs comme sur `absorbant/`).
- `raw_data_map.png` (**carte**) : heatmap 2D sortie vs (géométrie, axe), un panneau par canal.
  - **1 seule colonne géométrie** (`secteur1/`, `secteur1_amp_phase/`, `mesure_couplage/`, `mesure_gain/`) -> axe des géométries en unités physiques réelles, ex. **theta vs fréquence** pour `mesure_gain/`.
  - **Plusieurs colonnes géométrie** (`absorbant/`, `mesure_diag/`) -> pas d'axe physique unique possible, donc géométries triées par 1ère composante principale (même projection PCA que `00_visualize_data.py`).

```bash
cd IA_mesure/experiments/mesure_gain
python 06_plot_raw_data.py                 # comparison/raw_data_slices.png + raw_data_map.png
python 06_plot_raw_data.py --n-samples 10
```

**Labels d'axes et noms de colonnes trompeurs (`mesure_gain/`, `mesure_couplage/`)** : ces deux fichiers ont des noms de colonnes h5 inversés par rapport à leur vrai sens physique (voir la table "Correspondance dataset -> config" et les commentaires en tête de leurs `config.yaml`). `config.yaml` utilisait déjà le bon mapping de *données* (`axis_column`/`X_columns` pointent vers les bonnes colonnes h5), mais les graphiques affichaient encore le nom de colonne h5 brut comme libellé -- ex. l'axe contenant 2-11 GHz s'appelait "theta" et l'axe contenant -90°..90° s'appelait "freq", littéralement à l'envers de la réalité physique. Corrigé via une clé optionnelle `dataset.display_names` dans `config.yaml` (utilisée par `_common/metrics.py::display_name`, appliquée dans `_common/data_viz.py` et `_common/raw_data_viz.py`) :

```yaml
dataset:
  display_names:
    freq: "theta (deg)"
    theta: "freq (GHz)"
```

Sans cette clé (les 4 autres datasets), les libellés restent simplement les noms de colonnes h5 -- aucun changement de comportement pour eux.

## Plots/rapport standard (`generate_results`) après chaque entraînement

`01_train_gfr.py`, `02_train_active_gfr.py` (sur le checkpoint du dernier round) et `03_tune_hyperparams.py` (sur le meilleur essai, s'il y en a un) appellent automatiquement `surmod.generate_results()` (= `core/gen_results.py`) juste après l'entraînement. Ça relit le checkpoint + les données et sauvegarde, dans `results/.../<run>/plots/` :

- `train_val_convergence.png` -- courbe de perte train/val ;
- `amplitude_phase.png` -- seulement si la sortie est polaire/complexe (`gain_dB`/`phase`) ;
- `predictions_vs_true.png` -- vrai vs prédit pour chaque canal de sortie, sur un échantillon de géométries ;
- `training_report.txt` -- résumé texte (config, métriques, split).

Passez `--skip-report` à n'importe lequel des 3 scripts pour sauter cette étape (utile pour des essais très rapides où le rechargement des données n'apporte rien).

**Correctif apporté au passage (`core/gen_results.py`)** : le calcul de `gain_rmse_dB`/`phase_rmse_deg` ne se déclenchait que sur `n_out >= 2`, sans vérifier que la sortie est bien une quantité complexe (`common.complex: true`) -- pour `secteur1_amp_phase/` (sortie déjà en `amplitude_linear`/`phase_degrees`, pas en réel/imaginaire), ça calculait silencieusement un "Gain RMSE"/"Phase RMSE" n'importe quoi (vu en testant : 19.96 dB affiché, aberrant). La condition inclut maintenant `config["common"].get("complex", False)`.

## Bruit de mesure simulé (`absorbant/` uniquement)

`absorbant.h5` a l'air d'être une simulation propre (S_real/S_imag lisses en fréquence, écart-type ~0.47-0.48). Pour rendre le pipeline plus représentatif d'une vraie mesure (bruit d'instrument), un bruit gaussien additif est injecté sur `S_real`/`S_imag` **avant** toute conversion polaire, via le nouveau paramètre `common.measurement_noise_std` dans `DataLoader._process_data` (`src/surmod/core/data_loader.py`) :

```yaml
common:
  measurement_noise_std: 0.02   # ~ -34 dB par rapport à |S| <= 1 -- un niveau de bruit VNA plausible
```

- Ce paramètre est générique (utilisable pour n'importe quel dataset) mais **n'est activé que pour `absorbant/config.yaml`** -- les 5 autres configs n'ont pas cette clé, donc gardent leurs données propres (valeur par défaut : `0.0`, désactivé).
- Le bruit est identique pour `absorbant_gfr` et `absorbant_active_gfr` (défini dans le bloc `base` partagé par les deux), donc la comparaison baseline/active-learning reste équitable.
- Pour revenir aux données propres : mettre `measurement_noise_std: 0.0` (ou supprimer la ligne) dans `absorbant/config.yaml`.
- Le `DataLoader` affiche un message (`[DataLoader] Injected measurement noise...`) quand le bruit est appliqué, pour éviter toute confusion silencieuse.

## Limiter le CPU utilisé (`--cpus`, cluster partagé)

PyTorch utilise par défaut **tous les cœurs CPU disponibles sur la machine** pour ses calculs (matmul, etc.) -- ce n'est pas spécifique à Optuna ni à `03_tune_hyperparams.py` : `01_train_gfr.py`/`02_train_active_gfr.py` font pareil. Sur un poste perso c'est le comportement voulu ; sur un nœud de cluster partagé, un seul `python 03_tune_hyperparams.py` peut affamer tous les autres jobs.

Deux façons de le brider :

```bash
# 1) Sans toucher au code, via les variables d'environnement (marche tout de suite) :
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python 03_tune_hyperparams.py

# 2) Avec le flag --cpus (ajouté aux 3 scripts qui entrainent un modele) :
python 01_train_gfr.py --cpus 4
python 02_train_active_gfr.py --cpus 4
python 03_tune_hyperparams.py --cpus 4
```

`--cpus` appelle `torch.set_num_threads(N)` (+ `set_num_interop_threads(1)`) tout en haut du script, avant toute construction de `DataLoader`/modèle. Par défaut (sans `--cpus`), rien ne change : le comportement reste "utiliser tous les cœurs", pour ne pas surprendre quelqu'un qui tourne sur sa propre machine.

`run_all.py --jobs N` (N datasets en parallèle) fait déjà ce plafonnement automatiquement (répartit les cœurs entre les N jobs) -- ce `--cpus` couvre le cas où vous lancez un script directement, en dehors de `run_all.py`.

## Que fait `03_tune_hyperparams.py` exactement ? (exemple : `absorbant/`)

Déroulé complet d'un `python 03_tune_hyperparams.py` (voir `src/surmod/core/tuner.py::run_tuning`) :

1. **Charge `config.yaml`** pour l'expérience `<dataset>_gfr` (ex. `absorbant_gfr` = templates `base` + `model_gfr` + `train_standard` + `tuning_search`), applique les éventuels `--epochs`/`--patience-start`.
2. **Construit le `DataLoader` UNE SEULE FOIS** pour toute la recherche (pas rechargé à chaque essai) : lit `absorbant.h5` (124 000 lignes), reforme la grille (4000 géométries x 31 fréquences), **injecte le bruit de mesure** (`measurement_noise_std: 0.02`, une fois, avec le seed fixé -> les mêmes données bruitées pour tous les essais), convertit en polaire (gain_dB, sin, cos), normalise. Le split train(2560)/val(640)/test(800) est aussi fixé une fois pour toutes (même seed -> identique à chaque essai).
3. **Crée une étude Optuna** (`optuna.db` à la racine de `GFR_NET/`, stockage SQLite partagé entre tous les datasets) avec un `MedianPruner` (`n_startup_trials=5, n_warmup_steps=400, interval_steps=80`).
4. **Boucle sur les essais** (jusqu'à `--n-trials` ou `--time-hours`, le premier atteint) ; pour `absorbant_gfr`, chaque essai tire au sort (voir le bloc `tuning:` de `config.yaml`) :
   `hidden_dim` ∈ {128,256,384,512}, `n_fourier` ∈ {32,48,64,96}, `n_layers` ∈ [4,8], `dropout` ∈ [0,0.3], `lr` (log-uniforme) ∈ [1e-5,1e-2], `w_decay` (log-uniforme) ∈ [1e-7,1e-2], `batch_size` ∈ {512,1024,2048}.
   Les autres réglages du modèle (`n_geom_cont: 6`, `n_geo_fourier: 8`, `use_film: true`, ...) restent fixes, pris de `model_gfr`.
5. **Entraîne un `GFR_Net` frais** avec ces hyperparamètres sur le pool complet (2560 train / 640 val), jusqu'à `training.epochs` (400 par défaut pour `absorbant`) ou arrêt anticipé (`patience`), en sauvegardant le meilleur checkpoint de l'essai dans `results/absorbant/baseline_GFR_Net_<timestamp>_tuning/trials/trial_<N>/`.
6. **Reporte `val_loss` à Optuna à chaque époque** (`trial.report`) -- le `MedianPruner` peut couper un essai qui se comporte nettement moins bien que la médiane des essais précédents à la même époque, *une fois passées les `n_warmup_steps=400` premières époques*.
7. Quand un essai devient le **meilleur de l'étude**, son `config.yaml`/`train_metrics.csv`/checkpoint sont copiés à la racine de `results/absorbant/baseline_GFR_Net_<timestamp>_tuning/` (`best_run_info.json` y référence le chemin exact).
8. À la fin, imprime le meilleur essai (numéro, valeur, hyperparamètres).

**Point important pour `absorbant` spécifiquement** : `training.epochs` vaut **400**, exactement égal à `n_warmup_steps` du pruner. Un essai n'atteint donc jamais la fin de la période de warmup avant sa dernière époque (indices 0..399) -- **le pruning ne se déclenchera quasiment jamais avec la config par défaut**, même si le mécanisme fonctionne (voir le correctif du bug de pruning plus haut). Pour en profiter vraiment sur `absorbant`, montez `training.epochs` au-dessus de 400 (ou passez `--epochs 600` par exemple), ou réduisez `n_warmup_steps` (actuellement en dur dans `tuner.py`, pas exposé en config).

**Combien de temps / CPU ça prend, concrètement ?** Sur ce sandbox (CPU seulement, 4 cœurs Xeon 2.8GHz), entraîner `GFR_Net` sur `absorbant` avec la config par défaut (hidden_dim=384, batch_size=1024) prend environ **6-7 secondes par époque**. Avec `epochs=400` par essai :
- 1 essai complet ≈ 400 x 6.5s ≈ **43 minutes**.
- `03_tune_hyperparams.py` par défaut fait `--n-trials 10 --time-hours 3.0` -> **10 essais complets demanderaient ~7h**, largement plus que le plafond de 3h. En pratique, le plafond de temps s'arrêtera après ~4 essais complets, pas 10. Si vous voulez vraiment 10 essais complets, passez `--time-hours 8` (ou augmentez selon le nombre de cœurs réellement disponibles sur votre machine/cluster) ou réduisez `--epochs`.
- Consommation CPU : un seul process (`n_jobs=1` côté Optuna, essais strictement séquentiels), mais PyTorch sature plusieurs cœurs par défaut pendant l'entraînement (voir section précédente) -- utilisez `--cpus N` pour la limiter explicitement sur un cluster partagé.

## Visualiser l'optimisation Optuna (`05_visualize_tuning.py`)

```bash
cd IA_mesure/experiments/absorbant
python 05_visualize_tuning.py          # utilise la recherche la plus recente pour ce dataset
python 05_visualize_tuning.py --study-name GFR_Net_absorbant_absorbant_20260915_1200
```

Sauvegarde dans `comparison/` (via `_common/optuna_viz.py`) :
- `optuna_history.png` -- valeur de l'objectif par essai + meilleure valeur au fil du temps ;
- `optuna_importances.png` -- quels hyperparamètres influencent le plus le résultat ;
- `optuna_parallel_coordinate.png` -- coordonnées parallèles (voir les combinaisons gagnantes) ;
- `optuna_slice.png` -- un scatter par hyperparamètre (valeur testée vs objectif) ;
- `optuna_pruning.png` -- courbes val_loss par époque de chaque essai, et où ils ont été coupés (maintenant que le pruner fonctionne réellement, voir plus haut).

Toutes les études (tous datasets confondus) vivent dans le même fichier `GFR_NET/optuna.db` (SQLite) ; `05_visualize_tuning.py` retrouve automatiquement les études du bon dataset par leur nom.

**Pour une vue interactive** (recommandé si vous itérez beaucoup sur le tuning) : [`optuna-dashboard`](https://github.com/optuna/optuna-dashboard) tourne directement sur le même stockage SQLite, sans rien recalculer :

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///$(python -c "from surmod.utils.common import get_project_root; print(get_project_root())")/optuna.db
```

puis ouvrez `http://localhost:8080` (ou faites du port-forwarding si vous êtes sur un cluster distant).

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

### Un dataset à la fois

```bash
cd IA_mesure/experiments/secteur1
python 00_visualize_data.py                  # affiche le split train/val/test
python 01_train_gfr.py                       # baseline complet (config.yaml)
python 02_train_active_gfr.py                # active learning
python 03_tune_hyperparams.py --n-trials 30 --time-hours 4   # tuning Optuna
python 04_compare_models.py                  # tableau + graphes de comparaison
python 05_visualize_tuning.py                # graphes de la recherche Optuna
python 06_plot_raw_data.py                   # coupes + carte des données brutes
```

Chaque script accepte des options en ligne de commande pour des tests rapides sans toucher `config.yaml` (`--epochs`, `--n-rounds`, `--n-trials`, ...) -- voir `--help` ou l'en-tête de chaque fichier.

### Tous les datasets d'un coup : `run_all.py`

`run_all.py` (à la racine de `experiments/`) lance la pipeline pour les 6 dossiers automatiquement -- plus besoin de `cd` dans chacun à la main.

```bash
cd IA_mesure/experiments

# Baseline + active learning + comparaison, sur les 6 datasets (defaut ; le
# tuning Optuna n'est PAS inclus par defaut car c'est l'etape la plus lente)
python run_all.py

# Sanity check rapide avant un vrai run (epochs/rounds/trials reduits)
python run_all.py --quick

# Seulement certains datasets
python run_all.py --datasets secteur1 mesure_couplage

# Inclure aussi le tuning Optuna
python run_all.py --only train active tune compare

# Ne relancer que la comparaison (apres des trainings deja faits separement)
python run_all.py --only compare

# Paralleliser (attention a la charge CPU/RAM : N datasets entraines en meme temps)
python run_all.py --jobs 3

# S'arreter des la premiere erreur au lieu de continuer sur les autres datasets
python run_all.py --stop-on-error
```

Chaque étape de chaque dataset tourne dans son propre sous-processus : une erreur sur un dataset n'empêche pas les autres de continuer (sauf avec `--stop-on-error`). Avec `--jobs > 1`, le nombre de threads PyTorch par processus est automatiquement plafonné (`OMP_NUM_THREADS`/`MKL_NUM_THREADS`) pour éviter que N jobs ne se marchent dessus sur les mêmes cœurs CPU -- sans ça, `--jobs 3` sur une machine à 4 cœurs peut être **plus lent** que du séquentiel (observé : 8 min contre 22 s sur le même run une fois le plafonnement en place). Tout est journalisé dans `run_all_logs/<timestamp>/` :
- un fichier `<dataset>__<step>.log` par (dataset, étape) -- la sortie complète du script ;
- `all_datasets_comparison.csv` -- toutes les lignes de tous les `comparison_summary.csv` empilées (colonne `dataset` ajoutée), pour comparer les 6 jeux de données d'un coup d'œil ;
- un résumé (OK/FAIL + durée par étape) imprimé à la fin, avec un code de sortie non-nul si au moins une étape a échoué.

Sorties (non versionnées, voir `.gitignore`) :
```
<dataset>/results/<data_stem>/
  baseline_GFR_Net_<timestamp>/                 # 01_train_gfr.py
    plots/, training_report.txt                 #   via generate_results() (sauf --skip-report)
  active_active_gfr_net_active_r00/ ...         # 02_train_active_gfr.py (un dossier par round)
  active_active_learning/history.csv            # historique "précision vs nb labellisé"
                                                 #   plots/ generes pour le dernier round uniquement
  baseline_GFR_Net_<timestamp>_tuning/           # 03_tune_hyperparams.py (meilleur essai)
    plots/, training_report.txt                 #   idem, sur le meilleur essai
<dataset>/comparison/
  data_split_overview.png                       # 00_visualize_data.py
  comparison_summary.csv, comparison_loss.png, comparison_rmse.png,
  active_learning_efficiency.png                # 04_compare_models.py
  optuna_history.png, optuna_importances.png,
  optuna_parallel_coordinate.png, optuna_slice.png,
  optuna_pruning.png                            # 05_visualize_tuning.py
  raw_data_slices.png, raw_data_map.png         # 06_plot_raw_data.py
GFR_NET/optuna.db                               # SQLite storage shared by every dataset's Optuna study
```

## Limites connues / à garder en tête

- **`mesure_couplage.h5`** n'a que 16 géométries distinctes : l'active learning n'a de sens qu'à très petite échelle ici (pas un cas d'usage représentatif pour juger de l'efficacité de la méthode).
- Les hyperparamètres par défaut dans chaque `config.yaml` sont des points de départ raisonnables, pas le résultat d'une recherche -- utilisez `03_tune_hyperparams.py` pour les affiner sérieusement.
- Le tuning Optuna (`03_tune_hyperparams.py`) ne concerne que le `GFR_Net` baseline (le tuner ne gère pas encore l'active learning, cohérent avec le commentaire déjà présent dans `core/tuner.py`: "supervised only for now").
- Avec les budgets par défaut, `03_tune_hyperparams.py` sur `absorbant` ne complètera pas ses 10 essais dans les 3h par défaut (~43 min/essai) -- voir "Que fait `03_tune_hyperparams.py` exactement ?" pour les chiffres et comment ajuster.
- Le pruner Optuna (`MedianPruner`, `n_warmup_steps=400`) ne peut pas se déclencher sur `absorbant` avec `training.epochs: 400` (égal au warmup) -- augmentez `epochs` pour en profiter.
- Testé sur ce sandbox en CPU uniquement (pas de GPU) avec des runs très courts (quelques epochs / rounds / essais) juste pour valider que le pipeline tourne sans erreur de bout en bout sur les 6 datasets -- les valeurs numériques observées ne sont pas représentatives de la performance finale. Relancez avec les budgets par défaut (voir chaque `config.yaml`) pour de vrais résultats.

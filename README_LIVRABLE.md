# Examen DVC et DagsHub | Pipeline de régression

## Objectif
Prédire `silica_concentrate` à partir des paramètres d'une installation de flottation de minerai. Le jeu de données source contient 1 817 observations et 10 colonnes (incluant `date` et la cible).

## Méthodologie
1. `split`: lecture de `data/raw/raw.csv`, tri chronologique, séparation 80 % entraînement / 20 % test, sans mélange temporel. La date sert à l'ordonnancement, non comme variable explicative.
2. `normalize`: ajustement d'un `StandardScaler` uniquement sur l'entraînement et application au test. Conservation des versions normalisées et du scaler. Pour Random Forest, la normalisation n'est pas nécessaire ; le tuning et l'entraînement utilisent donc les variables non normalisées, afin d'éviter une fuite statistique entre les plis de validation croisée.
3. `grid_search`: `RandomForestRegressor`, grille de 8 combinaisons et validation chronologique `TimeSeriesSplit(3)`, sélection par RMSE.
4. `train`: apprentissage sur tout l'ensemble d'entraînement, paramètres issus de la recherche.
5. `evaluate`: mesure des performances sur le jeu de test chronologiquement postérieur ; calcul MAE, RMSE, R², export des prédictions.

## Reproduction
Depuis la racine du dépôt, après activation de `.venv` et installation des dépendances :

```powershell
python -m pip install dvc[s3] pandas numpy scikit-learn joblib
python -m pip check
dvc repro
dvc dag
dvc metrics show
dvc status
```

## Contrôles attendus
- Les quatre fichiers `X_train.csv`, `X_test.csv`, `y_train.csv` et `y_test.csv` sont générés.
- Les fichiers `X_train_scaled.csv`, `X_test_scaled.csv`, `models/data/scaler.joblib`, `models/models/best_params.pkl`, `models/models/trained_model.joblib`, `data/processed/predictions.csv` et `metrics/scores.json` existent après `dvc repro`.
- Les métriques sont calculées sur des observations non utilisées pendant l'entraînement ni la recherche d'hyperparamètres.
- Le graphe DVC comporte cinq étapes reliées.
- `dvc repro` relancé sans changement ne réexécute pas inutilement les étapes.

## Publication
Les scripts, `dvc.yaml`, `dvc.lock`, métriques et fichiers de configuration non secrets sont publiés avec Git. Les fichiers suivis par DVC sont envoyés avec `dvc push`. Ne jamais publier `.dvc/config.local`, les tokens, identifiants ou `.venv`.

## Limites
Le test chronologique constitue une évaluation hors échantillon sur une seule période. Le faible volume de données et un potentiel changement de régime du processus industriel limitent la généralisation. Le réentraînement périodique, le contrôle des dérives et des validations temporelles plus étendues seraient des prolongements pertinents.

## Transparence
La conception des scripts a bénéficié d'une assistance par IA. L'exécution, la vérification des résultats, la configuration des accès et la soumission relèvent de l'auteur du rendu. Les métriques ne sont pas présumées : elles doivent provenir de l'exécution effective du pipeline.

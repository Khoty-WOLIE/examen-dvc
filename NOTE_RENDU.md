# Note technique de rendu

**Sprint 17 - Suivi d’expérience et versioning**  
**Évaluation DVC et DagsHub**

## Objet

Ce rendu présente la mise en place d’un workflow de Machine Learning reproductible, avec DVC pour le suivi des données, des artefacts et des étapes de traitement, et DagsHub comme dépôt collaboratif et stockage distant.

## Travaux réalisés

Le travail couvre cinq étapes : séparation chronologique des données en ensembles d’entraînement et de test, normalisation ajustée sur l’entraînement, recherche d’hyperparamètres par validation croisée, entraînement d’un modèle de régression et évaluation sur l’ensemble de test. Les scripts sont organisés dans les répertoires du projet et orchestrés par la pipeline DVC.

## Résultats de l’exécution

L’exécution de la pipeline a abouti sur 1 817 observations, dont 1 453 pour l’entraînement et 364 pour le test. Les métriques observées sur le jeu de test sont : MAE = 0,8486 ; RMSE = 1,0429 ; R² = -0,2844.

## Interprétation et limites

Le R² négatif indique que, sur le jeu de test retenu, le modèle ne surpasse pas une prédiction constante fondée sur la moyenne des valeurs de test. Ce résultat est conservé sans présentation artificiellement favorable. L’objectif du travail est de démontrer une chaîne de traitement traçable et reproductible ; l’amélioration prédictive constitue une piste complémentaire, non une performance revendiquée.

## Vérifications avant soumission

La commande dvc repro a exécuté les étapes attendues et dvc status a indiqué que les données et la pipeline étaient à jour lors du contrôle local. La publication finale des artefacts DVC, leur consultation dans DagsHub et les droits d’accès du correcteur doivent être vérifiés avant soumission.

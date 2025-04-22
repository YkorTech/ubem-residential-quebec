# Système de Calibration

## Introduction

Le système de calibration du projet UBEM Résidentiel Québec utilise plusieurs approches complémentaires pour optimiser les paramètres des modèles énergétiques. Ces approches vont de l'analyse de sensibilité à la calibration hiérarchique multi-niveaux, en passant par la métamodélisation et l'apprentissage par transfert.

## Note sur les données de référence

**Important**: Les données de consommation Hydro-Québec utilisées pendant le développement de ce projet sont soumises à un accord de confidentialité et ne sont pas incluses dans le dépôt public.

Pour utiliser le système de calibration, vous devez:
- Fournir vos propres données de consommation d'énergie
- Respecter le format attendu:
  * Fichier CSV avec un minimum de deux colonnes:
    - `Intervalle15Minutes`: horodatages à intervalles de 15 minutes
    - `energie_sum_secteur`: valeurs de consommation énergétique
  * Le système convertira automatiquement les données 15 minutes en valeurs horaires (8760 heures)
  * Si vous avez déjà des données horaires, conservez le même format de colonnes
- Placer ces fichiers dans `data/inputs/hydro/conso_residentielle_YYYY.csv`

Le système de calibration compare la colonne `energie_sum_secteur` des données mesurées avec `Fuel Use: Electricity: Total` des résultats de simulation.

## Approches de calibration

### 1. Analyse de sensibilité

L'analyse de sensibilité utilise la méthode de Morris (elementary effects) pour identifier les paramètres les plus influents sur les résultats de simulation. Cette méthode est particulièrement efficace car elle:

- Nécessite relativement peu de simulations (~100 pour 10 paramètres)
- Identifie les effets non-linéaires et les interactions entre paramètres
- Calcule deux métriques importantes:
  * μ* (mu-star): Influence globale du paramètre
  * σ (sigma): Non-linéarité et interactions

### 2. Calibration par métamodélisation

La calibration par métamodélisation (`metamodel_calibrator.py`) construit un modèle approximatif (surrogate model) qui est rapide à évaluer et permet de remplacer les simulations complètes pendant l'optimisation:

- Design d'expériences (DOE) par Latin Hypercube Sampling pour une couverture efficace de l'espace des paramètres
- Construction d'un métamodèle au choix:
  * Gaussian Process Regression (GPR/Krigeage): Grande précision avec estimation de l'incertitude
  * Random Forest (RF): Robustesse pour les relations complexes et non-linéaires
- Optimisation bayésienne sur le métamodèle:
  * Exploration intelligente de l'espace des paramètres
  * Équilibre entre exploitation (des régions prometteuses) et exploration (de nouvelles régions)
- Validation avec simulations complètes des solutions optimales

### 3. Calibration hiérarchique multi-niveaux

La calibration hiérarchique (`hierarchical_calibrator.py`) décompose le processus de calibration en niveaux successifs, du global au spécifique:

- **Niveau 1 - Global Envelope**: Paramètres globaux d'enveloppe (infiltration_rate, wall_rvalue, ceiling_rvalue, window_ufactor)
- **Niveau 2 - Systems**: Paramètres des systèmes (heating_efficiency, heating_setpoint, cooling_setpoint)
- **Niveau 3 - Schedules**: Paramètres des schedules (occupancy_scale, lighting_scale, appliance_scale, temporal_diversity)

Avantages de cette approche:
- Réduction de la complexité à chaque niveau
- Propagation des résultats entre niveaux
- Temps de calibration réduit (~2-3 heures vs ~4-5 heures)
- Meilleure convergence pour les paramètres complexes

**Important**: L'ordre des niveaux doit être respecté. Les paramètres de schedules (niveau 3) dépendent des paramètres d'enveloppe (niveau 1).

### 4. Apprentissage par transfert

L'apprentissage par transfert (`transfer_learning_manager.py`) exploite les résultats des calibrations précédentes pour accélérer les nouvelles calibrations:

- Stockage des résultats dans une base de connaissances persistante
- Caractérisation du contexte de chaque calibration (année, zone, climat, etc.)
- Prédiction des paramètres initiaux optimaux pour de nouvelles calibrations
- Construction de modèles prédictifs à partir des calibrations antérieures
- Accélération progressive du processus (jusqu'à 50-75% de gain de temps)

## Paramètres calibrés

Les principaux paramètres calibrés sont:

| Paramètre | Description | Limites typiques |
|-----------|-------------|-----------------|
| infiltration_rate | Taux d'infiltration d'air | [-0.3, 0.3] |
| wall_rvalue | Résistance thermique des murs | [-0.3, 0.3] |
| ceiling_rvalue | Résistance thermique du plafond | [-0.3, 0.3] |
| window_ufactor | Facteur U des fenêtres | [-0.3, 0.3] |
| heating_efficiency | Efficacité du chauffage | [-0.3, 0.3] |
| heating_setpoint | Point de consigne chauffage | [-0.2, 0.2] |
| cooling_setpoint | Point de consigne climatisation | [-0.2, 0.2] |
| occupancy_scale | Facteur d'échelle occupation | [-0.3, 0.3] |
| lighting_scale | Facteur d'échelle éclairage | [-0.3, 0.3] |
| appliance_scale | Facteur d'échelle appareils | [-0.3, 0.3] |
| temporal_diversity | Diversité temporelle | [0.0, 1.0] |

Les valeurs représentent des ajustements relatifs (±30%) par rapport aux valeurs de base des archétypes.

## Métriques d'évaluation

Le système utilise plusieurs métriques pour évaluer la qualité de la calibration:

- **RMSE** (Root Mean Square Error): Mesure globale de l'erreur
- **MAE** (Mean Absolute Error): Erreur absolue moyenne
- **MAPE** (Mean Absolute Percentage Error): Erreur en pourcentage
- **Métriques saisonnières**: RMSE par saison
- **Métriques de pointe**: RMSE durant les périodes de pointe

## Utilisation

### Ligne de commande

```bash
# Analyse de sensibilité
python -m src.main sensitivity --year 2023 --trajectories 10

# Calibration par métamodélisation
python -m src.main metamodel --year 2023 --doe-size 100 --metamodel gpr

# Calibration hiérarchique
python -m src.main hierarchical --year 2023 --parameters all

# Calibration avec apprentissage par transfert
python -m src.main transfer --year 2023 --doe-size 50
```

### Fonctionnement interne

1. **Préparation**:
   - Chargement des données de référence (vos données de consommation)
   - Initialisation des paramètres à calibrer
   - Création du répertoire de sortie

2. **Design d'expériences**:
   - Génération des points par Latin Hypercube Sampling
   - Simulation de chaque point du design
   - Agrégation des résultats et calcul des métriques

3. **Construction du métamodèle**:
   - Normalisation des données
   - Entraînement du GPR ou RF
   - Validation du métamodèle

4. **Optimisation**:
   - Recherche des paramètres optimaux via optimisation bayésienne
   - Validation des meilleures solutions avec simulations complètes
   - Analyse des résultats et importance des paramètres

5. **Finalisation**:
   - Simulation finale avec les paramètres optimaux
   - Génération des graphiques et rapports
   - Stockage dans la base de connaissances (pour l'apprentissage par transfert) 
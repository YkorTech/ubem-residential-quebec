# UBEM Résidentiel Québec - Documentation technique

## Vue d'ensemble

UBEM Résidentiel Québec est un modèle énergétique urbain pour le secteur résidentiel du Québec utilisant une approche archétypale bottom-up avec calibration efficace et scientifiquement rigoureuse. Le système permet de simuler la consommation énergétique horaire des bâtiments résidentiels et d'appliquer différentes méthodes de calibration pour améliorer la précision des prédictions.

## Architecture du système

Le système est composé de plusieurs modules interconnectés:

1. **Module principal** (`main.py`): Point d'entrée et interface de ligne de commande
2. **Configuration** (`config.py`): Paramètres et chemins centralisés
3. **Gestion des scénarios** (`scenario_manager.py`): Transformation des archétypes pour les scénarios futurs
4. **Génération de schedules** (`schedule_generator.py`): Profils stochastiques d'occupation et d'usage
5. **Utilitaires** (`utils.py`): Fonctions communes et outils partagés
6. **Gestionnaires spécialisés** (dossier `managers/`):
   - `archetype_manager.py`: Conversion des archétypes vers le format HPXML
   - `simulation_manager.py`: Exécution des simulations OpenStudio/EnergyPlus
   - `aggregation_manager.py`: Classification des bâtiments et calcul des facteurs d'échelle
   - `calibration_manager.py`: Orchestration des processus de calibration
7. **Calibration** (dossier `calibration/`):
   - Analyse de sensibilité (méthode de Morris)
   - Calibration par métamodélisation (GPR, Random Forest)
   - Calibration hiérarchique multi-niveaux
   - Apprentissage par transfert
8. **Dashboard** (dossier `dashboard/`): Interface utilisateur web avec Plotly/Dash

## Intégration avec OpenStudio-HPXML

Le projet s'appuie sur [OpenStudio-HPXML](https://github.com/NREL/OpenStudio-HPXML), développé par NREL, pour transformer les archétypes en modèles EnergyPlus. L'`ArchetypeManager` joue un rôle central dans cette intégration:

- Conversion des paramètres d'archétypes (CSV) en format HPXML
- Transformation des unités métriques en unités impériales (m² → ft², RSI → R-value, etc.)
- Génération des workflows OpenStudio (`.osw`) pour chaque simulation
- Application des paramètres de calibration aux modèles HPXML

Pour plus de détails sur cette intégration cruciale, consultez la [documentation spécifique sur HPXML](hpxml_integration.md).

## Flux de travail typique

1. **Configuration**: Définition des paramètres et chemins
2. **Préparation des archétypes**: Conversion CSV → HPXML via l'ArchetypeManager
3. **Génération des schedules**: Création de profils d'usage (standard ou stochastiques)
4. **Simulation**: Exécution parallèle des simulations OpenStudio/EnergyPlus
5. **Agrégation**: Classification des bâtiments et calcul des facteurs d'échelle
6. **Calibration**: Amélioration de la précision via différentes méthodes
7. **Validation**: Comparaison avec les données Hydro-Québec
8. **Visualisation**: Affichage des résultats dans le dashboard

## Structure des données

### Entrées
- **Archétypes**: Caractéristiques des bâtiments types (`data/inputs/archetypes/`)
- **Météo**: Fichiers EPW par zone climatique (`data/inputs/weather/`)
- **Hydro-Québec**: Données de consommation réelle (`data/inputs/hydro/`)
  * **Note importante**: Les données originales d'Hydro-Québec ne sont pas fournies en raison d'un accord de confidentialité
  * Format attendu:
    - Fichier CSV avec colonnes `Intervalle15Minutes` (horodatages) et `energie_sum_secteur` (consommation)
    - Le système convertira automatiquement les données 15 minutes en valeurs horaires
  * Le système compare la colonne `energie_sum_secteur` des données mesurées avec `Fuel Use: Electricity: Total` des simulations
- **Évaluation foncière**: Classification et représentativité (`data/inputs/evaluation/`)

### Sorties
- **Simulations**: Résultats horaires et sommaires (`data/outputs/simulations/`)
- **Calibration**: Analyses de sensibilité et paramètres optimaux (`data/outputs/calibration/`)
- **Base de connaissances**: Données pour l'apprentissage par transfert (`data/outputs/knowledge_base/`)

## Configuration OpenStudio
- **Templates**: Workflows EnergyPlus (`data/openstudio/templates/`)
- **Mesures**: Scripts de transformation (`data/openstudio/measures/`)

## Fonctionnalités principales

1. **Conversion HPXML** des archétypes avec transformations d'unités appropriées
2. **Simulation parallèle** d'archétypes avec OpenStudio-HPXML/EnergyPlus
3. **Schedules stochastiques** pour l'occupation et l'usage énergétique
4. **Analyse de sensibilité** pour identifier les paramètres influents
5. **Calibration multi-approches** (métamodélisation, hiérarchique, apprentissage par transfert)
6. **Scénarios futurs** pour modéliser les projections climatiques et énergétiques
7. **Dashboard interactif** pour visualiser et analyser les résultats
8. **Agrégation rigoureuse** basée sur la représentativité des archétypes 
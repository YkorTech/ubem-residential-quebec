# Sources de données - UBEM Résidentiel Québec

Ce document décrit les différentes sources de données utilisées dans le projet UBEM Résidentiel Québec, ainsi que leur structure et organisation.

## Archétypes de bâtiments

### Source et origine

Les archétypes de bâtiments utilisés dans ce projet sont basés sur la bibliothèque développée par Ressources Naturelles Canada (RNCan), disponible sur [canmet-energy/housing-archetypes](https://github.com/canmet-energy/housing-archetypes). Cette bibliothèque comprend des archétypes résidentiels représentatifs du parc immobilier québécois, construits à partir d'études terrain et de données statistiques.

### Fichiers inclus

Le dépôt contient deux fichiers principaux d'archétypes:

1. **`base_archetypes.csv`**
   - Contient l'ensemble complet des 1644 archétypes du Québec
   - Inclut toutes les variantes de bâtiments par période de construction, typologie, et caractéristiques techniques
   - Utilisé comme référence complète et pour les analyses exhaustives

2. **`selected_archetypes.csv`**
   - Sous-ensemble de 66 archétypes sélectionnés pour le modèle
   - Représente un équilibre entre précision et performance de calcul
   - Les archétypes ont été sélectionnés pour représenter adéquatement la diversité du parc immobilier tout en optimisant les ressources de calcul
   - Utilisé par défaut dans les simulations standard

### Structure des données

Chaque fichier d'archétype contient les colonnes suivantes:
- Identifiants uniques et métadonnées (ID, année de construction, type de bâtiment, etc.)
- Caractéristiques géométriques (superficie, nombre d'étages, etc.)
- Propriétés thermiques (isolation des murs, toits, fondations, etc.)
- Systèmes mécaniques (chauffage, climatisation, ventilation, eau chaude)
- Caractéristiques des fenêtres et portes
- Infiltration d'air et étanchéité
- Paramètres d'efficacité énergétique

Ces données sont utilisées par l'`ArchetypeManager` pour générer les fichiers HPXML requis pour les simulations avec OpenStudio.

## Fichiers météo

### Organisation des fichiers

Les fichiers météo sont organisés en deux catégories principales:

1. **Fichiers historiques** (`data/inputs/weather/historic/`)
   - **EPW**: Contient les fichiers météo au format EPW (EnergyPlus Weather) pour chaque station représentative
   - **Mapping**: Le fichier `h2k_epw_avec_zones_meteo.csv` associe chaque zone météo du Québec à la station météo correspondante
   - Ces fichiers sont basés sur les données climatiques historiques et sont utilisés pour les simulations en conditions actuelles

2. **Fichiers futurs** (`data/inputs/weather/future/`)
   - Organisés en sous-dossiers selon les scénarios climatiques:
     * `warm/`: Scénario de réchauffement élevé
     * `typical/`: Scénario de réchauffement moyen
     * `cold/`: Scénario de réchauffement faible
   - Le fichier `zones_futur_fichiers_epw.csv` fournit le mapping entre les zones climatiques et les fichiers EPW futurs
   - Ces fichiers permettent de simuler l'impact du changement climatique sur la consommation énergétique des bâtiments

### Source des données météo

- Les fichiers EPW historiques sont dérivés des données d'Environnement Canada
- Les fichiers EPW futurs sont générés à partir de modèles climatiques et de projections de changement climatique

## Données d'évaluation foncière

### Fichier principal

Le fichier `data/inputs/evaluation/2024.csv` contient les données d'évaluation foncière utilisées pour:
- Déterminer la distribution des types de bâtiments par zone
- Calculer les facteurs d'échelle pour l'extrapolation à l'échelle régionale
- Établir les caractéristiques démographiques et spatiales du parc immobilier

### Structure des données

Ce fichier contient des informations sur:
- L'emplacement géographique (municipalité, code postal, zone)
- La typologie des bâtiments
- L'année de construction
- Les dimensions et caractéristiques physiques
- La valeur foncière
- Et d'autres attributs pertinents pour la modélisation énergétique

## Templates de workflow OpenStudio

### Fichiers de template

Les templates de workflow OpenStudio sont essentiels pour configurer les simulations et sont stockés dans `data/openstudio/templates/`:

1. **`base_workflow.osw`**
   - Template standard pour les simulations déterministes
   - Configure les mesures OpenStudio de base et les paramètres par défaut
   - Utilisé pour les simulations de référence

2. **`stochastic_workflow.osw`**
   - Template pour les simulations avec profils d'occupation stochastiques
   - Inclut les mesures pour générer des schedules aléatoires basés sur des données statistiques
   - Permet de représenter la variabilité des comportements des occupants

### Mesures et composants

Le répertoire `data/openstudio/measures/` contient les mesures OpenStudio personnalisées développées pour le projet, qui peuvent être référencées dans les templates de workflow.

## Données de consommation pour calibration

Pour utiliser le module de calibration, vous devez fournir des données de consommation énergétique:

- Format: Fichier CSV nommé `conso_residentielle_YYYY.csv` (où YYYY est l'année)
- Emplacement: `data/inputs/hydro/`
- Structure requise:
  * Colonne `Intervalle15Minutes`: Horodatage à intervalle de 15 minutes
  * Colonne `energie_sum_secteur`: Consommation énergétique en kWh
- Le système convertira les données 15 minutes en valeurs horaires pour la comparaison avec les résultats de simulation

Ces données sont essentielles pour la calibration du modèle mais ne sont pas incluses dans le dépôt en raison de contraintes de confidentialité. Les utilisateurs doivent fournir leurs propres données dans le format spécifié. 
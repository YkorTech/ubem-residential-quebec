# Scénarios Futurs (2035/2050)

## Introduction

Le module de scénarios futurs permet de transformer les archétypes pour simuler différentes projections climatiques et énergétiques à l'horizon 2035 et 2050. Le système supporte 24 scénarios basés sur une nomenclature structurée qui combine les dimensions principales.

## Nomenclature des scénarios

Les scénarios futurs sont nommés selon une nomenclature systématique:

```
{année}_{type_climat}_{croissance_électrification}_{efficacité}
```

Par exemple: `2035_warm_PV_E` désigne un scénario pour 2035, avec climat chaud, croissance standard (Predicted Value) et efficacité standard.

## Dimensions des scénarios

Chaque scénario combine 4 dimensions, créant un total de 24 scénarios possibles (2×3×2×2):

### 1. Horizon temporel (2 options)

- **2035**: Horizon moyen terme
- **2050**: Horizon long terme

### 2. Conditions climatiques (3 options)

- **warm**: Année chaude
  * 2035: +0.5°C par rapport à la référence
  * 2050: +1.0°C par rapport à la référence
- **typical**: Année normale
  * 2035: +1.0°C par rapport à la référence
  * 2050: +2.0°C par rapport à la référence
- **cold**: Année froide 
  * 2035: +1.5°C par rapport à la référence
  * 2050: +3.0°C par rapport à la référence

### 3. Croissance et électrification (2 options)

- **PV**: Predicted Value (standard)
  * 2035: 95% chauffage électrique, +10% croissance du parc
  * 2050: 100% chauffage électrique, +25% croissance du parc
- **UB**: Upper Boundary (accéléré)
  * 2035: 100% chauffage électrique, +12% croissance du parc
  * 2050: 100% chauffage électrique, +30% croissance du parc

### 4. Amélioration de l'efficacité (2 options)

- **E**: Efficacité standard
  * 2035: +30% isolation, -30% infiltration, COP 3.5
  * 2050: +50% isolation, -50% infiltration, COP 4.0
- **EE**: Efficacité maximale
  * 2035: +50% isolation, -50% infiltration, COP 4.0
  * 2050: +100% isolation, -70% infiltration, COP 5.0

## Exemples de scénarios

- `2035_warm_PV_E`: 2035, année chaude, croissance standard, efficacité standard
- `2050_cold_UB_EE`: 2050, année froide, croissance accélérée, efficacité maximale
- `2035_typical_UB_E`: 2035, année normale, croissance accélérée, efficacité standard
- `2050_typical_PV_EE`: 2050, année normale, croissance standard, efficacité maximale

## Implémentation (`scenario_manager.py`)

Le `ScenarioManager` est le composant central pour la gestion des scénarios futurs. Il offre plusieurs fonctionnalités clés:

### 1. Transformation des archétypes

La méthode `transform_archetype` applique les paramètres du scénario à un archétype:
- Amélioration de l'enveloppe (isolation, infiltration)
- Électrification des systèmes de chauffage
- Mise à jour des efficacités (COP, HSPF)
- Adaptation des paramètres de confort (points de consigne)

### 2. Transformation des données foncières

La méthode `transform_property_data` modifie les données d'évaluation foncière pour refléter:
- Croissance du parc immobilier
- Évolution des typologies (densification)
- Distribution des années de construction pour les nouveaux bâtiments

### 3. Adaptation des paramètres calibrés

La méthode `apply_scenario_to_calibration_params` ajuste les paramètres calibrés pour:
- Éviter la double application des améliorations d'enveloppe
- Adapter intelligemment les paramètres aux nouvelles efficacités
- Préserver les relations entre paramètres

## Utilisation des scénarios

### Simulation d'un scénario

Pour simuler un scénario futur spécifique:

```bash
python -m src.main simulate --future-scenario 2035_warm_PV_E
```

### Fichiers météo

Chaque scénario utilise des fichiers météo spécifiques:

- Pour les scénarios `warm`, les fichiers sont dans `data/inputs/weather/future/warm/`
- Pour les scénarios `typical`, les fichiers sont dans `data/inputs/weather/future/typical/`
- Pour les scénarios `cold`, les fichiers sont dans `data/inputs/weather/future/cold/`

Le système sélectionne automatiquement les fichiers appropriés en fonction du scénario.

## Cas d'utilisation

Les scénarios futurs permettent d'explorer:

1. **Impact du changement climatique**: Comparer les scénarios `warm`, `typical` et `cold` pour un même horizon
2. **Effet des politiques d'efficacité**: Comparer les scénarios `E` et `EE` pour mesurer l'impact des mesures d'efficacité
3. **Trajectoires d'électrification**: Comparer les scénarios `PV` et `UB` pour évaluer différentes vitesses d'électrification
4. **Évolution temporelle**: Comparer 2035 et 2050 pour observer les tendances à long terme

## Notebook dédié

Un notebook Jupyter dédié `simulate_future_scenarios.ipynb` est disponible pour:
- Simuler individuellement chaque scénario
- Exécuter tous les scénarios en batch
- Analyser les résultats
- Comparer différents scénarios
- Visualiser les résultats avec des graphiques
- Exporter les résultats pour des analyses externes 
# Générateur de Schedules Stochastiques

## Introduction

Le générateur de schedules stochastiques (`schedule_generator.py`) permet de créer des profils d'occupation et d'usage énergétique personnalisés pour chaque archétype. Ces profils temporels représentent les comportements variables des occupants et l'utilisation des équipements dans les bâtiments. Contrairement aux schedules standards (fixes), ils introduisent une variabilité réaliste qui améliore la précision des simulations.

## Types de schedules générés

Le système génère trois types principaux de schedules stochastiques:

### 1. Profils d'occupation

Représentent la présence des occupants dans les bâtiments:
- Variation jour/nuit
- Différenciation semaine/weekend
- Adaptés au nombre d'occupants
- Différents niveaux d'activité (éveillé actif, éveillé passif, dormant)

### 2. Profils d'utilisation de l'éclairage

Représentent l'utilisation de l'éclairage:
- Corrélés avec les profils d'occupation
- Variations saisonnières (plus d'éclairage en hiver)
- Variations en fonction de l'heure du jour

### 3. Profils d'utilisation des appareils électroménagers

Représentent l'utilisation des équipements électriques:
- Appareils de cuisine (cooking_range)
- Lave-vaisselle (dishwasher)
- Machine à laver (clothes_washer)
- Téléviseur (plug_loads_tv)
- Autres appareils (plug_loads_other)
- Ventilateurs de plafond (ceiling_fan)

## Paramètres de calibration spécifiques

Les schedules stochastiques peuvent être ajustés via plusieurs paramètres:

- **`occupancy_scale`**: Facteur d'échelle pour les profils d'occupation [-0.3, 0.3]
- **`lighting_scale`**: Facteur d'échelle pour les profils d'éclairage [-0.3, 0.3]
- **`appliance_scale`**: Facteur d'échelle pour les profils d'appareils [-0.3, 0.3]
- **`temporal_diversity`**: Diversité temporelle entre archétypes [0.0, 1.0]

Ces paramètres permettent d'ajuster les schedules lors de la calibration pour mieux correspondre aux données mesurées.

## Approche de génération

### Profils journaliers

Les profils journaliers (24h) sont générés en combinant:

- **Profils de base**: Modèles typiques pour chaque usage
- **Styles comportementaux**: Différents types de comportements
- **Variations aléatoires**: Pour refléter l'imprévisibilité

Styles comportementaux implémentés:
- `early_riser`: Lève-tôt, activité matinale
- `late_sleeper`: Couche-tard, activité en soirée
- `standard`: Comportement standard, horaires habituels
- `home_worker`: Travail à domicile, présence forte en journée
- `night_owl`: Noctambule, activité nocturne

### Profils annuels (8760h)

Les profils annuels sont construits en:

- Combinant les profils journaliers pour chaque jour de l'année
- Différenciant les jours de semaine et de weekend
- Ajoutant des variations saisonnières pour certains usages
- Appliquant des décalages temporels pour diversifier les archétypes

### Diversité temporelle

La diversité temporelle est un concept clé qui permet d'éviter que tous les archétypes suivent exactement les mêmes horaires:

- À `temporal_diversity = 0.0`: Tous les archétypes utilisent leurs appareils aux mêmes heures
- À `temporal_diversity = 1.0`: Les archétypes peuvent avoir des décalages importants (±2.5h)

Les décalages sont appliqués différemment selon la période de la journée:
- Matin (5h-10h): Décalage de ±2.5h max
- Soir (16h-21h): Décalage de ±2.5h max
- Mi-journée (10h-16h): Décalage de ±1.5h max
- Nuit (21h-5h): Décalage de ±1.0h max

### Caractéristiques importantes

- Génération déterministe basée sur les caractéristiques de l'archétype (ID, surface, etc.)
- Reproductibilité des résultats pour la même graine aléatoire
- Format horaire compatible avec EnergyPlus
- Adaptation automatique au nombre de jours dans le mois

## Utilisation dans le projet

### Activation des schedules stochastiques

Pour utiliser les schedules stochastiques dans une simulation:

```bash
python -m src.main simulate --year 2023 --stochastic
```

### API Python

```python
from src.schedule_generator import ScheduleGenerator

# Initialiser le générateur
generator = ScheduleGenerator()

# Profils à générer
profile_types = ['occupants', 'lighting_interior', 'lighting_garage', 
                'cooking_range', 'dishwasher', 'clothes_washer']

# Générer et sauvegarder un schedule
schedule_path = generator.generate_and_save_schedule(
    profile_types=profile_types,
    archetype_id=123,
    archetype_data={
        'weather_zone': 448,
        'houseType': 'single-family detached',
        'vintageExact': 1985,
        'spaceHeatingType': 'electric resistance',
        'numAdults': 2,
        'numChildren': 1
    },
    scale_factors={
        'occupancy_scale': 0.1,
        'lighting_scale': -0.05,
        'appliance_scale': 0.15
    },
    temporal_diversity=0.5,
    output_dir='path/to/output'
)
```

### Visualisation dans le dashboard

Le dashboard inclut un onglet "Schedules" permettant de visualiser:
- Les profils générés pour chaque archétype
- Les patterns d'occupation journaliers
- La diversité entre archétypes
- Les variations saisonnières

### Stockage des schedules générés

Les schedules générés sont sauvegardés dans:
- `data/inputs/schedules/[année]/[archetype_id]/`

Ces fichiers sont référencés dans les workflows OpenStudio lors de la simulation.

## Avantages des schedules stochastiques

1. **Réalisme accru**: Meilleure représentation des comportements réels des occupants
2. **Diversité**: Évite les pics artificiels causés par des schedules identiques
3. **Calibration**: Permet d'ajuster les patterns d'usage pour la calibration
4. **Précision**: Améliore généralement la correspondance avec les données mesurées

## Limitations

1. **Temps de calcul**: Légèrement plus long que les simulations avec schedules standards
2. **Complexité**: Introduit plus de paramètres à calibrer
3. **Variabilité**: Peut introduire une légère variabilité dans les résultats

## Exemple de workflow typique

1. Exécuter une simulation initiale avec schedules standards
2. Lancer une simulation avec schedules stochastiques
3. Comparer les résultats
4. Calibrer les paramètres de schedules (`occupancy_scale`, etc.)
5. Relancer la simulation avec les paramètres optimisés 
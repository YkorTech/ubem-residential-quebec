# Intégration avec OpenStudio-HPXML

## Introduction

Le projet UBEM Résidentiel Québec s'appuie sur [OpenStudio-HPXML](https://github.com/NREL/OpenStudio-HPXML), un puissant framework développé par le National Renewable Energy Laboratory (NREL) pour la modélisation énergétique des bâtiments résidentiels. Cette intégration est essentielle pour transformer les données d'archétypes en modèles énergétiques simulables.

## Qu'est-ce qu'OpenStudio-HPXML?

OpenStudio-HPXML est un ensemble de mesures OpenStudio qui permettent:
- La simulation de bâtiments résidentiels en utilisant un fichier HPXML comme description du bâtiment
- La conversion automatique de cette description vers un modèle EnergyPlus
- Une vaste gamme de capacités de modélisation (systèmes HVAC, enveloppe, schedules, etc.)
- Des calculs de dimensionnement automatisés (autosizing)
- Des générations de résultats standardisés

## Le rôle de l'ArchetypeManager

L'`ArchetypeManager` (`src/managers/archetype_manager.py`) est le composant central qui gère cette intégration. Ses principales fonctions sont:

1. **Chargement des archétypes**: Lecture du fichier CSV `selected_archetypes.csv` pour obtenir les caractéristiques des bâtiments types
2. **Conversion des paramètres**: Transformation des paramètres d'archétypes en attributs HPXML compatibles
3. **Conversion d'unités**: Passage du système métrique (SI) au système impérial pour EnergyPlus
4. **Génération des workflows**: Création des fichiers de configuration OpenStudio (`.osw`) pour chaque archétype
5. **Application des paramètres de calibration**: Intégration des paramètres optimisés dans les workflows

## Processus de conversion

### 1. Définition des mappings

L'ArchetypeManager définit un ensemble complet de mappings qui associent les colonnes du fichier d'archétypes aux paramètres HPXML:

```python
self.mappings = {
    'year_built': HPXMLMapping(
        hpxml_name='year_built',
        archetype_name='vintageExact',
        conversion_func=self.converter.convert_vintage
    ),
    'geometry_unit_cfa': HPXMLMapping(
        hpxml_name='geometry_unit_cfa',
        archetype_name='totFloorArea',
        conversion_func=self.converter.m2_to_ft2
    ),
    # ... et bien d'autres mappings
}
```

### 2. Conversions d'unités

Des convertisseurs spécifiques sont implémentés pour les transformations nécessaires:

- `m2_to_ft2`: Conversion des surfaces de m² à pi²
- `rsi_to_rvalue`: Conversion des résistances thermiques de RSI à R-value
- `cop_to_efficiency`: Conversion des COP en SEER/EER/HSPF selon le type d'équipement
- `c_to_f`: Conversion des températures de Celsius à Fahrenheit
- Et plusieurs autres conversions spécialisées

### 3. Classification des types de bâtiments

L'ArchetypeManager inclut un `BuildingTypeMapper` qui détermine le type de bâtiment HPXML approprié:

- `single-family detached`: Maisons unifamiliales détachées
- `single-family attached`: Maisons jumelées ou en rangée
- `apartment unit`: Appartements
- `manufactured home`: Maisons mobiles/préfabriquées

### 4. Gestion des particularités pour les thermopompes

Un traitement spécial est appliqué pour les systèmes de thermopompes:
- Conversion des COP en HSPF pour les thermopompes air-air
- Conservation des COP pour les thermopompes géothermiques
- Définition des types d'efficacité appropriés (HSPF/SEER pour air-air, COP/EER pour géothermiques)

## Préparation d'un archétype pour simulation

La méthode principale `prepare_archetype` orchestre le processus complet:

1. Récupération des données de l'archétype
2. Détermination de la configuration HPXML du bâtiment
3. Initialisation des paramètres HPXML de base
4. Application des paramètres de calibration si disponibles
5. Application des conversions et mappings pour tous les attributs
6. Création du workflow OpenStudio
7. Génération des schedules stochastiques si demandé

## Génération des workflows OpenStudio

Les workflows sont générés à partir de templates (`.osw`) et personnalisés pour chaque archétype:

1. Le template approprié est sélectionné (`base_workflow.osw` ou `stochastic_workflow.osw`)
2. Les arguments sont nettoyés et mis à jour avec les paramètres HPXML de l'archétype
3. Le fichier météo approprié est associé en fonction de la zone climatique
4. Le workflow est sauvegardé dans le répertoire de l'archétype pour exécution ultérieure

## Intégration avec les scénarios futurs

Pour les scénarios futurs, l'ArchetypeManager travaille en coordination avec le `ScenarioManager`:

1. Le `ScenarioManager` transforme d'abord l'archétype selon les paramètres du scénario
2. L'ArchetypeManager utilise ensuite cet archétype transformé pour créer le workflow HPXML
3. Des vérifications supplémentaires sont effectuées pour assurer la cohérence des types d'efficacité

## Intégration avec la calibration

Lors de la calibration, l'ArchetypeManager:
1. Reçoit un dictionnaire de paramètres calibrés pour chaque archétype
2. Applique ces paramètres ajustés avant de générer les workflows
3. Gère intelligemment l'interaction entre paramètres calibrés et paramètres standards
4. Applique des contraintes pour maintenir des valeurs physiquement réalistes

## Exemple d'utilisation dans le code

```python
from src.managers.archetype_manager import ArchetypeManager

# Initialiser le gestionnaire d'archétypes
archetype_manager = ArchetypeManager()

# Préparer un archétype pour simulation
arch_dir = archetype_manager.prepare_archetype(
    archetype_id=123,
    output_dir=Path('data/outputs/simulations/2023/baseline'),
    year=2023,
    calibration_params={
        'infiltration_rate': {123: 0.15},  # +15% d'infiltration
        'wall_rvalue': {123: -0.10}        # -10% d'isolation des murs
    },
    use_stochastic_schedules=True
) 
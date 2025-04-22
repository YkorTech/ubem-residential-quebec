# Guide d'installation - UBEM Résidentiel Québec

Ce guide décrit les étapes nécessaires pour installer et configurer l'environnement de développement pour le projet UBEM Résidentiel Québec.

## Prérequis

- Python 3.8+ (recommandé: Python 3.10)
- OpenStudio 3.6+ avec support HPXML
- 16 GB RAM minimum (32 GB recommandé)
- 50 GB d'espace disque pour les données et résultats

## Composantes principales

### OpenStudio-HPXML

Ce projet utilise [OpenStudio-HPXML](https://github.com/NREL/OpenStudio-HPXML), un framework développé par NREL qui permet la modélisation énergétique de bâtiments résidentiels via:
- La description standardisée des bâtiments en format HPXML
- Un workflow automatisé pour la génération des modèles EnergyPlus
- Des capacités avancées de simulation (autosizing, schedules stochastiques, etc.)

L'intégration avec OpenStudio-HPXML est gérée principalement par le module `ArchetypeManager` qui:
- Convertit les paramètres des archétypes du format CSV en paramètres HPXML
- Applique les conversions d'unités nécessaires (SI vers Imperial)
- Configure les workflows OpenStudio pour chaque simulation

## Note importante sur les données

**Concernant les données de consommation énergétique**:
- Les données Hydro-Québec originales utilisées dans le développement de ce projet sont soumises à un accord de confidentialité et ne peuvent pas être partagées publiquement.
- Le système de calibration est néanmoins entièrement fonctionnel si vous disposez de vos propres données de consommation horaire annuelle au format similaire.
- Format requis pour les données de calibration:
  * Fichier CSV avec un minimum de deux colonnes:
    - `Intervalle15Minutes`: horodatages à intervalles de 15 minutes
    - `energie_sum_secteur`: valeurs de consommation énergétique
  * Le système convertira automatiquement les données 15 minutes en valeurs horaires
  * Si vous avez déjà des données horaires, assurez-vous qu'elles ont le même format de colonne
- Placez vos données dans `data/inputs/hydro/conso_residentielle_YYYY.csv` où YYYY est l'année de référence.

## Installation pas à pas

### 1. Cloner le dépôt

```bash
# HTTPS
git clone https://github.com/YkorTech/ubem-residential-quebec.git

# OU SSH
git clone git@github.com:YkorTech/ubem-residential-quebec.git

cd ubem-residential-quebec
```

### 2. Créer un environnement virtuel

```bash
# Avec venv
python -m venv venv

# Activer l'environnement
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Installer OpenStudio

1. Télécharger OpenStudio 3.6.1 (ou plus récent) depuis [le site officiel](https://www.openstudio.net/downloads)
2. Suivre les instructions d'installation pour votre système d'exploitation
3. S'assurer que la variable d'environnement `OPENSTUDIO_EXE` est correctement configurée

> **Important**: OpenStudio-HPXML est inclus dans OpenStudio 3.6+. Pour les versions antérieures, vous devrez l'installer séparément depuis le [dépôt GitHub de NREL](https://github.com/NREL/OpenStudio-HPXML).

### 5. Configuration des chemins

Le fichier `src/config.py` contient les chemins et paramètres de configuration du projet. Ce fichier est déjà inclus dans le dépôt avec des paramètres par défaut.

Si nécessaire, modifiez ce fichier pour:
- Vérifier que `BASE_DIR` pointe correctement vers votre répertoire d'installation
- Ajuster les chemins vers OpenStudio selon votre installation
- Configurer le niveau de parallélisation avec `CPU_USAGE` (par défaut: 0.8 = 80% des CPU)
- S'assurer que les chemins vers les templates OpenStudio-HPXML sont corrects

### 6. Structure des données

La structure de dossiers nécessaire est déjà incluse dans le dépôt Git. Lorsque vous clonez le projet, tous les dossiers requis sont automatiquement créés avec leur hiérarchie complète. Vous n'avez pas besoin de créer manuellement les dossiers.

Les dossiers principaux sont:
- `data/inputs/archetypes` - Pour les fichiers d'archétypes
- `data/inputs/weather` - Pour les fichiers météorologiques
- `data/inputs/hydro` - Pour les données de consommation énergétique
- `data/inputs/evaluation` - Pour les données d'évaluation foncière
- `data/inputs/schedules` - Pour les fichiers de schedules générés
- `data/openstudio` - Pour les templates et mesures OpenStudio
- `data/outputs` - Pour les résultats des simulations

### 7. Données de base

Pour que le système fonctionne, vous devez fournir:

1. **Archétypes** (`data/inputs/archetypes/`):
   - Le dépôt inclut deux fichiers principaux:
     * `base_archetypes.csv`: L'ensemble complet des 1644 archétypes du Québec
     * `selected_archetypes.csv`: Un sous-ensemble de 66 archétypes sélectionnés pour le modèle
   - Ces archétypes sont basés sur la bibliothèque de [Ressources Naturelles Canada](https://github.com/canmet-energy/housing-archetypes)
   - Par défaut, le modèle utilise `selected_archetypes.csv` pour optimiser les performances

2. **Fichiers météo** (`data/inputs/weather/`):
   - Fichiers historiques (`historic/`):
     * Fichiers EPW pour chaque zone climatique dans `historic/epw/`
     * Fichier de mapping `historic/h2k_epw_avec_zones_meteo.csv` qui associe zones et fichiers
   - Fichiers futurs (`future/`):
     * Sous-dossiers pour les types de climat (`warm/`, `typical/`, `cold/`)
     * Fichier de mapping `zones_futur_fichiers_epw.csv` pour les scénarios futurs

3. **Données foncières** (`data/inputs/evaluation/`):
   - Fichier `2024.csv` avec les données d'évaluation foncière
   - Ce fichier est utilisé pour le calcul des facteurs d'échelle

4. **Données de consommation** (`data/inputs/hydro/`):
   - Format requis: `conso_residentielle_YYYY.csv` avec colonnes `Intervalle15Minutes` et `energie_sum_secteur`
   - Le système comparera la colonne `energie_sum_secteur` des données mesurées avec `Fuel Use: Electricity: Total` des simulations
   - Ces données sont essentielles pour la calibration, mais vous devez fournir vos propres données
   - Si vous n'avez pas de données de consommation, la simulation fonctionne mais pas la calibration

5. **Templates OpenStudio** (`data/openstudio/templates/`):
   - `base_workflow.osw` pour les simulations standards
   - `stochastic_workflow.osw` pour les simulations avec schedules stochastiques

Ces templates sont essentiels pour la configuration des workflows OpenStudio-HPXML.

## Documentation

Pour plus de détails, consultez la documentation complète:

- [Index de documentation](docs/index.md) - Point d'entrée central de la documentation
- [Sources de données](docs/core/data_sources.md) - Description des fichiers de données
- [Guide d'utilisation du dashboard](docs/usage/dashboard.md) - Interface utilisateur
- [Guide des commandes](docs/usage/command_line.md) - Référence des commandes disponibles

## Lancement rapide

Une fois l'installation terminée, vous pouvez:

```bash
# Lancer le dashboard
python -m src.main dashboard

# Exécuter une simulation simple
python -m src.main simulate --year 2023 --stochastic

# Simuler un scénario futur
python -m src.main simulate --future-scenario 2035_warm_PV_E
```

## Dépannage

### OpenStudio ou HPXML non trouvé

Si vous recevez une erreur concernant OpenStudio ou HPXML:

1. Vérifier que le chemin dans `config.py` est correct
2. S'assurer que la variable d'environnement `OPENSTUDIO_EXE` est définie correctement
3. Vérifier la version d'OpenStudio (3.6+ pour HPXML intégré)
4. Sur Windows, ajouter le répertoire d'installation d'OpenStudio au PATH

### Erreur de mémoire

Si vous rencontrez des erreurs de mémoire:

1. Réduire le nombre de simulations parallèles dans `config.py`:
   ```python
   CPU_USAGE = 0.5  # Réduire à 50% des CPUs
   ```

2. Libérer de la mémoire en fermant d'autres applications
3. Augmenter la mémoire virtuelle/swap si possible

### Fichiers manquants

Si certains fichiers ne sont pas trouvés:

1. Vérifier les chemins dans `config.py`
2. S'assurer que les fichiers d'entrée sont placés aux bons endroits
3. Consulter les logs pour identifier les fichiers manquants spécifiques

## Support

Pour obtenir de l'aide, veuillez:
1. Consulter la documentation dans le dossier `docs/`
2. Vérifier les problèmes connus dans les Issues GitHub
3. Créer une nouvelle Issue si nécessaire 
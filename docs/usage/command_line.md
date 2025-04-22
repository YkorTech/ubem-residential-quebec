# Guide d'utilisation - Ligne de commande

Ce document présente les commandes principales disponibles dans le projet UBEM Résidentiel Québec.

## Commandes de base

```bash
# Format général
python -m src.main [COMMANDE] [OPTIONS]
```

## Commandes de simulation

### Simulation standard

```bash
# Simulation avec horaires standards pour une année spécifique
python -m src.main simulate --year 2023
```

### Simulation avec horaires stochastiques

```bash
# Simulation avec horaires stochastiques (plus réalistes)
python -m src.main simulate --year 2023 --stochastic
```

### Simulation de scénarios futurs

```bash
# Simulation d'un scénario futur spécifique
python -m src.main simulate --future-scenario 2035_warm_PV_E

# Simulation d'un scénario futur avec plus de détails dans les logs
python -m src.main simulate --future-scenario 2050_cold_PV_E --debug
```

## Commandes de calibration

### Analyse de sensibilité

```bash
# Exécuter une analyse de sensibilité (méthode de Morris)
python -m src.main sensitivity --year 2023 --trajectories 10
```

### Calibration par métamodélisation

```bash
# Calibration complète par métamodélisation
python -m src.main metamodel --year 2023 --doe-size 100 --metamodel gpr

# Calibration avec paramètres spécifiques
python -m src.main metamodel --year 2023 --parameters infiltration_rate,wall_rvalue,ceiling_rvalue
```

### Calibration hiérarchique

```bash
# Calibration hiérarchique multi-niveaux
python -m src.main hierarchical --year 2023 --parameters all
```

### Calibration avec apprentissage par transfert

```bash
# Calibration avec apprentissage par transfert (accélérée)
python -m src.main transfer --year 2023 --doe-size 50

# Gestion de la base de connaissances
python -m src.main knowledge --discover
```

## Lancement du dashboard

```bash
# Lancer l'interface utilisateur web
python -m src.main dashboard
```

## Paramètres communs

| Paramètre | Description | Exemple |
|-----------|-------------|---------|
| `--year` | Année de simulation | `--year 2023` |
| `--stochastic` | Utiliser des horaires stochastiques | `--stochastic` |
| `--future-scenario` | Scénario futur à simuler | `--future-scenario 2035_warm_PV_E` |
| `--doe-size` | Taille du design d'expériences | `--doe-size 100` |
| `--metamodel` | Type de métamodèle (gpr ou rf) | `--metamodel gpr` |
| `--parameters` | Paramètres spécifiques à calibrer | `--parameters infiltration_rate,wall_rvalue` |
| `--debug` | Mode de débogage avec logs détaillés | `--debug` |

## Exemples d'utilisation

### Workflow typique de calibration

```bash
# 1. Exécuter l'analyse de sensibilité
python -m src.main sensitivity --year 2023 --trajectories 10

# 2. Calibrer avec les paramètres les plus influents
python -m src.main hierarchical --year 2023 --parameters all

# 3. Simuler avec les paramètres calibrés
python -m src.main simulate --year 2023 --stochastic
```

### Workflow de scénarios futurs

```bash
# 1. Calibrer sur l'année de référence
python -m src.main hierarchical --year 2023 --parameters all

# 2. Simuler tous les scénarios futurs
# Exemple pour 2035 (à répéter pour les différents scénarios)
python -m src.main simulate --future-scenario 2035_warm_PV_E
python -m src.main simulate --future-scenario 2035_typical_PV_E
python -m src.main simulate --future-scenario 2035_cold_PV_E
# etc.
``` 
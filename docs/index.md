# Documentation UBEM Résidentiel Québec

Bienvenue dans la documentation du projet UBEM Résidentiel Québec. Cette documentation est organisée en plusieurs sections pour vous aider à naviguer et comprendre le projet.

## Structure de la documentation

### Guides principaux
- [Guide d'installation](../INSTALLATION.md) - Comment installer et configurer le projet
- [README](../README.md) - Vue d'ensemble du projet

### Documentation technique
- [Architecture du système](core/overview.md) - Structure et composants du système
- [Sources de données](core/data_sources.md) - Description des données d'entrée et leur structure

### Fonctionnalités
- [Système de calibration](calibration/overview.md) - Documentation sur les approches de calibration
- [Schedules stochastiques](schedules/overview.md) - Génération et utilisation des profils d'occupation stochastiques
- [Scénarios futurs](scenarios/overview.md) - Modélisation et analyse des scénarios 2035/2050

### Guides d'utilisation
- [Dashboard](usage/dashboard.md) - Utilisation de l'interface utilisateur interactif
- [Ligne de commande](usage/command_line.md) - Guide complet des commandes et options disponibles

## Arborescence des fichiers principaux

```
data/
├── inputs/                    # Données d'entrée
│   ├── archetypes/           # Base d'archétypes (base et selected)
│   ├── evaluation/           # Données d'évaluation foncière
│   ├── hydro/                # Données de consommation (à fournir)
│   ├── schedules/            # Fichiers de schedules générés
│   └── weather/              # Fichiers météo (historiques et futurs)
├── openstudio/               # Configuration OpenStudio
└── outputs/                  # Résultats des simulations

src/
├── calibration/              # Modules de calibration
├── dashboard/                # Interface utilisateur web
├── managers/                 # Gestionnaires des composantes
└── *.py                      # Modules principaux

docs/
├── core/                     # Documentation technique
├── calibration/              # Documentation sur la calibration
├── schedules/                # Documentation sur les schedules
│   └── overview.md           # Guide complet des schedules stochastiques
├── scenarios/                # Documentation sur les scénarios futurs
│   └── overview.md           # Guide complet des scénarios futurs
└── usage/                    # Guides d'utilisation
    ├── dashboard.md          # Utilisation du dashboard
    └── command_line.md       # Référence des commandes
```

## Fichiers exclus

Certains fichiers ne sont pas inclus dans le dépôt pour des raisons de taille ou de confidentialité:

- Données de consommation réelles (`data/inputs/hydro/*.csv`)
- Fichiers météo (`*.epw`)
- Données d'évaluation foncière (`data/inputs/evaluation/*.csv`)
- Résultats de simulation (`data/outputs/`)

Les utilisateurs doivent fournir leurs propres données de consommation pour la calibration.
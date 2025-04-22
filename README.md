# UBEM Résidentiel Québec

Un modèle énergétique urbain (UBEM) pour le secteur résidentiel du Québec utilisant une approche archétypale bottom-up avec calibration efficace et scientifiquement rigoureuse.

## Documentation du projet

La documentation complète est disponible dans les sections suivantes:

- [Guide d'installation](INSTALLATION.md) - Instructions détaillées pour installer et configurer le projet
- [Documentation détaillée](docs/index.md) - Point d'entrée central vers toute la documentation technique
- [Sources de données](docs/core/data_sources.md) - Description des fichiers de données et leur organisation
- [Documentation technique](docs/core/overview.md) - Architecture et composants du système
- [Système de calibration](docs/calibration/overview.md) - Approches de calibration et paramètres
- [Scénarios futurs](docs/scenarios/overview.md) - Documentation sur les scénarios 2035/2050

## Interface utilisateur

Le projet comprend un dashboard technique interactif permettant de:
- Paramétrer de façon avancée les simulations (années, scénarios futurs, schedules)
- Configurer et exécuter les différentes méthodes de calibration (sensibilité, métamodèle, hiérarchique)
- Visualiser les résultats et les métriques de performance
- Comparer différents scénarios et configurations

![Dashboard UBEM Résidentiel Québec](docs/usage/ubem_residence_dashboard.png)

Pour plus de détails sur l'utilisation du dashboard technique, consultez la [documentation dédiée](docs/usage/dashboard.md).

## État d'avancement du projet

✅ **Phase 1: Analyse de sensibilité + Métamodélisation** - Implémentée
- Analyse de sensibilité (méthode de Morris)
- Calibration par métamodélisation (GPR et RF)

✅ **Phase 2: Calibration hiérarchique** - Implémentée
- Approche multi-niveaux pour la calibration
- Propagation des résultats entre niveaux

✅ **Phase 3: Apprentissage par transfert** - Implémentée
- Base de connaissances des calibrations
- Prédiction des paramètres initiaux

✅ **Phase 4: Méthode d'agrégation rigoureuse** - Implémentée
- Approche basée sur la représentativité
- Élimination du raisonnement circulaire

✅ **Phase 5: Scénarios futurs (2035/2050)** - Implémentée
- 24 scénarios combinant climat, croissance, électrification et efficacité
- Transformation automatique des archétypes selon les scénarios
- Support des fichiers météo futurs
- Application automatique des paramètres de calibration aux scénarios futurs
- Notebook dédié à l'analyse des scénarios futurs

⏳ **Phase 6: Automatisation et interface** - Planifiée
- Workflows automatisés
- Interface utilisateur complète

## Description

Ce projet vise à simuler la consommation énergétique du secteur résidentiel au Québec en utilisant :
- Simulation annuelle avec pas de temps horaire
- OpenStudio/EnergyPlus via OpenStudio-HPXML
- Calibration avancée basée sur analyse de sensibilité, métamodélisation, calibration hiérarchique et apprentissage par transfert
- Agrégation rigoureuse basée uniquement sur la représentativité des archétypes
- Comparaison avec les données de consommation réelles
- Simulation de scénarios futurs (2035, 2050)
- Support des horaires standards et stochastiques

Ce projet a été développé dans le cadre d'un Projet Intégrateur 4 à Polytechnique Montréal en partenariat avec Hydro-Québec. Il représente le composant résidentiel d'un projet académique plus large visant à modéliser l'ensemble de la demande électrique du Québec (UBEM complet).

## Structure du Projet

```
├── data/
│   ├── inputs/                    # Données d'entrée
│   │   ├── archetypes/           # Base d'archétypes (base et selected)
│   │   ├── evaluation/           # Données d'évaluation foncière
│   │   ├── hydro/                # Données de consommation (à fournir)
│   │   ├── schedules/            # Fichiers de schedules générés
│   │   └── weather/              # Fichiers météo EPW (historiques et futurs)
│   ├── openstudio/               # Configuration OpenStudio
│   │   ├── measures/             # Mesures OpenStudio
│   │   └── templates/            # Templates de workflow
│   └── outputs/                  # Résultats des simulations
│       ├── simulations/          # Résultats des simulations brutes
│       ├── calibration/          # Résultats des calibrations
│       └── knowledge_base/       # Base de connaissances pour l'apprentissage par transfert
├── docs/                         # Documentation détaillée
│   ├── core/                     # Architecture et composants
│   ├── calibration/              # Documentation sur la calibration
│   ├── schedules/                # Documentation sur les schedules stochastiques
│   ├── scenarios/                # Documentation sur les scénarios futurs
│   └── usage/                    # Guides d'utilisation
└── src/                          # Code source
    ├── calibration/              # Modules de calibration
    ├── dashboard/                # Interface utilisateur web
    ├── managers/                 # Gestionnaires des différentes composantes
    └── *.py                      # Modules principaux
```

## Prérequis

- Python 3.8+ (recommandé: Python 3.10)
- OpenStudio 3.6+ (avec support HPXML)
- NumPy 2.0+ (pour la sérialisation JSON améliorée)
- 16 GB RAM minimum (32 GB recommandé)
- Espace disque : 50 GB minimum pour les données et résultats

## Utilisation

Consultez le [guide d'installation](INSTALLATION.md) pour configurer le projet, puis utilisez les commandes suivantes:

```bash
# Lancer le dashboard
python -m src.main dashboard

# Exécuter une simulation simple
python -m src.main simulate --year 2023 --stochastic

# Simuler un scénario futur
python -m src.main simulate --future-scenario 2035_warm_PV_E
```

## License

MIT License. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Contact

- GitHub: [YkorTech](https://github.com/YkorTech)
- Email: ykortech@gmail.com
- LinkedIn: [olivier-youfang-kamgang](https://linkedin.com/in/olivier-youfang-kamgang)
- Site web: [ykortech.com](https://ykortech.com)
- Pour toute question ou collaboration, n'hésitez pas à ouvrir une issue sur ce dépôt.
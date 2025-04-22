"""
Module d'apprentissage par transfert pour UBEM Québec.
Exploite les calibrations précédentes pour accélérer les futures calibrations.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import glob
import os
import time
import tempfile 
import atexit  
import platform
import random
import shutil

from ..config import config
from ..utils import ensure_dir, save_results

class FileLock:
    """Classe de verrouillage de fichier portable qui fonctionne sur Windows et Unix."""
    
    def __init__(self, path, timeout=10):
        self.path = Path(path)
        self.lockfile = self.path.with_suffix('.lock')
        self.timeout = timeout
        self.locked = False
    
    def acquire(self):
        """Acquérir le verrou avec timeout."""
        start_time = time.time()
        
        # Générer un ID unique pour ce processus
        pid = os.getpid()
        rand = random.randint(0, 1000000)
        lock_id = f"{pid}-{rand}-{datetime.now().strftime('%H%M%S')}"
        
        # Boucle d'essai avec timeout
        while time.time() - start_time < self.timeout:
            try:
                # Essayer de créer le fichier de verrou
                if not self.lockfile.exists():
                    # Créer un fichier temporaire et le déplacer atomiquement
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=self.path.parent)
                    temp_file.write(lock_id)
                    temp_file.close()
                    
                    # Déplacer le fichier pour atomicité
                    try:
                        shutil.move(temp_file.name, self.lockfile)
                        self.locked = True
                        return True
                    except:
                        # Si le déplacement échoue, quelqu'un d'autre a créé le fichier entre-temps
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)
                
                # Si on est là, soit le fichier existe déjà, soit on n'a pas pu le créer
                # Attendre un peu et réessayer
                time.sleep(0.1)
            except Exception as e:
                # Erreur lors de la tentative d'acquisition
                # Attendre un peu plus longtemps
                time.sleep(0.5)
        
        # Timeout
        return False
    
    def release(self):
        """Libérer le verrou."""
        if self.locked and self.lockfile.exists():
            try:
                os.unlink(self.lockfile)
                self.locked = False
                return True
            except:
                return False
        return True
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

class TransferLearningManager:
    """Gère l'apprentissage par transfert entre calibrations successives."""
    
    # Variables de classe pour le Singleton
    _instance = None
    _kb_loaded = False
    
    def __new__(cls, *args, **kwargs):
        # Création d'une seule instance partagée
        if cls._instance is None:
            cls._instance = super(TransferLearningManager, cls).__new__(cls)
            cls._instance.logger = logging.getLogger("src.calibration.transfer_learning_manager")
            cls._instance._models_cache = {}
            cls._instance.knowledge_base = None
            cls._instance.transfer_model = None
            cls._instance.scaler_X = StandardScaler()
            cls._instance.scaler_y = StandardScaler()
            cls._instance.db_path = config.paths['output'].ROOT / 'knowledge_base'
            ensure_dir(cls._instance.db_path)
            
            # Nouveau : Indicateurs de dernier chargement
            cls._instance.model_file = cls._instance.db_path / 'transfer_model.pkl'
            cls._instance.kb_file = cls._instance.db_path / 'knowledge_base.pkl'
            cls._instance.last_model_update = 0
            cls._instance.last_kb_update = 0
            
            # S'assurer que les fichiers sont nettoyés à la fin
            atexit.register(cls._instance._cleanup)
            
        return cls._instance
    
    def _cleanup(self):
        """Nettoie les ressources à la fin du processus."""
        # Nettoyer les fichiers de verrou qui pourraient avoir été oubliés
        for lockfile in self.db_path.glob("*.lock"):
            try:
                if lockfile.exists():
                    os.unlink(lockfile)
            except:
                pass
    
    def __init__(self):
        """
        Initialisation minimaliste qui effectue uniquement le chargement de la base
        de connaissances si nécessaire.
        """
        # Ne charger la base qu'une seule fois pour toutes les instances
        if not TransferLearningManager._kb_loaded:
            # NOUVEAU: Vérifier si nous sommes dans un processus enfant avec préchargement
            if os.environ.get('TRANSFER_LEARNING_PRELOADED') == 'TRUE':
                # Dans un processus enfant après préchargement, éviter de recharger
                self.logger.debug("Utilisation des modèles préchargés dans le processus parent")
                TransferLearningManager._kb_loaded = True
            else:
                # Processus normal, charger la base
                self._load_knowledge_base()
                TransferLearningManager._kb_loaded = True
    
    def _load_knowledge_base(self) -> None:
        """Charge la base de connaissances des calibrations précédentes."""
        # Vérifier si le fichier KB a été modifié depuis le dernier chargement
        try:
            if self.kb_file.exists():
                last_modified = os.path.getmtime(self.kb_file)
                if last_modified <= self.last_kb_update and self.knowledge_base is not None:
                    # Le fichier n'a pas changé, pas besoin de recharger
                    return
                
                # Essayer d'acquérir le verrou
                with FileLock(self.kb_file) as lock:
                    if lock.locked:
                        # Vérifier à nouveau après avoir acquis le verrou
                        last_modified = os.path.getmtime(self.kb_file)
                        if last_modified <= self.last_kb_update and self.knowledge_base is not None:
                            return
                        
                        with open(self.kb_file, 'rb') as f:
                            self.knowledge_base = pickle.load(f)
                        self.last_kb_update = last_modified
                        self.logger.info(f"Base de connaissances chargée avec {len(self.knowledge_base)} entrées")
                        
                        # Mettre à jour le modèle de transfert si suffisamment de données et s'il n'est pas déjà chargé
                        if len(self.knowledge_base) >= 3 and self.transfer_model is None:
                            self._load_transfer_model()
                    else:
                        # Si on ne peut pas acquérir le verrou, on utilise ce qu'on a déjà si disponible
                        if self.knowledge_base is None:
                            self.knowledge_base = []
                            self.logger.warning("Impossible d'acquérir le verrou pour charger la KB, initialisation vide")
            else:
                self.logger.info("Aucune base de connaissances existante trouvée, création d'une nouvelle base")
                self.knowledge_base = []
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la base de connaissances: {str(e)}")
            self.knowledge_base = []
    
    def _load_transfer_model(self) -> None:
        """Charge le modèle de transfert s'il existe."""
        try:
            if self.model_file.exists():
                last_modified = os.path.getmtime(self.model_file)
                if last_modified <= self.last_model_update and self.transfer_model is not None:
                    # Le modèle n'a pas changé, pas besoin de recharger
                    return
                
                self.logger.info("Chargement du modèle de transfert existant")
                with open(self.model_file, 'rb') as f:
                    self.transfer_model = pickle.load(f)
                self.last_model_update = last_modified
                self.logger.info("Modèle de transfert chargé avec succès")
                return True
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle de transfert: {str(e)}")
        return False
    
    def _save_knowledge_base(self) -> None:
        """Sauvegarde la base de connaissances."""
        lock = FileLock(self.kb_file)
        if not lock.acquire():
            self.logger.error("Impossible d'acquérir le verrou pour sauvegarder la KB")
            return
        
        try:
            # Créer un fichier temporaire pour éviter la corruption
            temp_file = tempfile.NamedTemporaryFile(delete=False, dir=self.db_path)
            temp_path = Path(temp_file.name)
            temp_file.close()
            
            with open(temp_path, 'wb') as f:
                pickle.dump(self.knowledge_base, f)
            
            # Remplacer le fichier original de manière sécurisée
            if platform.system() == 'Windows':
                # Windows nécessite de supprimer d'abord le fichier existant
                if self.kb_file.exists():
                    os.unlink(self.kb_file)
            
            os.rename(temp_path, self.kb_file)
            self.last_kb_update = os.path.getmtime(self.kb_file)
            
            self.logger.info(f"Base de connaissances sauvegardée avec {len(self.knowledge_base)} entrées")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la base de connaissances: {str(e)}")
            # Nettoyer le fichier temporaire si nécessaire
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        finally:
            lock.release()
    
    def _extract_context_features(self, year: int, 
                                 scenario: str = 'baseline',
                                 use_stochastic_schedules: bool = False) -> Dict:
        """
        Extrait les caractéristiques du contexte de calibration.
        
        Args:
            year: Année de calibration
            scenario: Scénario de calibration
            use_stochastic_schedules: Utilisation d'horaires stochastiques
            
        Returns:
            Dictionnaire des caractéristiques de contexte
        """
        # Caractéristiques de base
        features = {
            'year': year,
            'is_stochastic': int(use_stochastic_schedules),
            'is_baseline': int(scenario == 'baseline'),
            'is_future': int(scenario in ['2035', '2050']),
        }
        
        
        
        return features
    
    def _update_transfer_model(self) -> None:
        """Met à jour le modèle de transfert avec les données actuelles."""
        if len(self.knowledge_base) < 3:
            self.logger.warning("Pas assez de données pour entraîner un modèle de transfert")
            return
        
        # Créer une clé unique pour ce modèle
        cache_key = f"model_{len(self.knowledge_base)}"
        
        # Vérifier si ce modèle existe déjà en cache
        if cache_key in self._models_cache:
            self.logger.info(f"Utilisation du modèle en cache pour {len(self.knowledge_base)} entrées")
            self.transfer_model = self._models_cache[cache_key]
            return
        
        # Vérifier si un autre processus est en train de mettre à jour le modèle
        lock = FileLock(self.model_file)
        if not lock.acquire():
            # Si on ne peut pas acquérir le verrou, essayer de charger le modèle existant
            self.logger.info("Un autre processus met à jour le modèle, tentative de chargement du modèle existant")
            if self._load_transfer_model():
                return
            else:
                self.logger.warning("Impossible de charger le modèle existant, utilisation du modèle actuel")
                return
            
        try:
            # Vérifier si le modèle a été mis à jour par un autre processus entre-temps
            if self.model_file.exists():
                last_modified = os.path.getmtime(self.model_file)
                if last_modified > self.last_model_update:
                    # Le modèle a été mis à jour, le charger
                    self.logger.info("Un modèle plus récent a été détecté, chargement...")
                    with open(self.model_file, 'rb') as f:
                        self.transfer_model = pickle.load(f)
                    self.last_model_update = last_modified
                    lock.release()
                    return
            
            # Extraire les caractéristiques et les paramètres optimaux
            contexts = []
            optimal_params = []
            
            for entry in self.knowledge_base:
                # Normaliser les noms de paramètres (supprimer les préfixes période)
                param_dict = {}
                for key, value in entry['optimal_params'].items():
                    # Extraire le nom de base du paramètre (supprimer période_)
                    if '_' in key:
                        base_name = key.split('_')[-1]  # Prendre la partie après le dernier _
                        param_dict[base_name] = value
                    else:
                        param_dict[key] = value
                
                contexts.append(entry['context'])
                optimal_params.append(param_dict)
            
            # Identifier les paramètres communs à toutes les entrées
            common_params = set(optimal_params[0].keys())
            for params in optimal_params[1:]:
                common_params = common_params.intersection(params.keys())
            
            # Si aucun paramètre commun, impossible de construire un modèle cohérent
            if not common_params:
                self.logger.warning("Pas de paramètres communs entre les calibrations")
                lock.release()
                return
                
            self.logger.info(f"Construction du modèle avec {len(common_params)} paramètres communs")
            
            # Construire matrices de caractéristiques et cibles
            X = []
            y_dict = {param: [] for param in common_params}
            
            for i, entry in enumerate(self.knowledge_base):
                # Caractéristiques de contexte à liste plate
                features = list(contexts[i].values())
                X.append(features)
                
                # Paramètres optimaux
                for param in common_params:
                    y_dict[param].append(optimal_params[i].get(param, 0.0))
            
            X = np.array(X)
            
            # Standardiser les caractéristiques
            X_scaled = self.scaler_X.fit_transform(X)
            
            # Créer un modèle pour chaque paramètre
            self.transfer_model = {}
            
            for param in common_params:
                y = np.array(y_dict[param])
                
                # Standardiser les cibles
                y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
                
                # Créer et entraîner le modèle de Random Forest
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=42
                )
                
                # Entraîner sur toutes les données
                model.fit(X_scaled, y_scaled)
                
                # Stocker le modèle, les scalers et les scores CV
                self.transfer_model[param] = {
                    'model': model,
                    'scaler_X': self.scaler_X,
                    'scaler_y': self.scaler_y,
                    'importance': model.feature_importances_
                }
                
                self.logger.info(f"Modèle pour {param} entraîné avec succès")
            
            # Sauvegarder le modèle atomiquement
            temp_file = tempfile.NamedTemporaryFile(delete=False, dir=self.db_path)
            temp_path = Path(temp_file.name)
            temp_file.close()
            
            with open(temp_path, 'wb') as f:
                pickle.dump(self.transfer_model, f)
            
            # Remplacer le fichier original de manière sécurisée
            if platform.system() == 'Windows':
                # Windows nécessite de supprimer d'abord le fichier existant
                if self.model_file.exists():
                    os.unlink(self.model_file)
                    
            os.rename(temp_path, self.model_file)
            self.last_model_update = os.path.getmtime(self.model_file)

            # Stocker le modèle en cache
            self._models_cache[cache_key] = self.transfer_model
            self.logger.info(f"Modèle mis en cache avec la clé {cache_key}")
                    
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour du modèle de transfert: {str(e)}")
            # Nettoyer le fichier temporaire si nécessaire
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        finally:
            lock.release()
    
    def predict_optimal_parameters(self, parameters: Dict[str, Dict],
                                 year: int,
                                 scenario: str = 'baseline',
                                 use_stochastic_schedules: bool = False) -> Dict[str, float]:
        """
        Prédit les paramètres optimaux pour un nouveau contexte.
        
        Args:
            parameters: Dictionnaire des paramètres à calibrer avec leurs bornes
            year: Année cible
            scenario: Scénario cible
            use_stochastic_schedules: Utilisation d'horaires stochastiques
            
        Returns:
            Dictionnaire des valeurs prédites pour les paramètres
        """
        # Créer une clé unique pour ce contexte de prédiction
        context_key = f"{year}_{scenario}_{use_stochastic_schedules}"
        
        # Vérifier si on a déjà fait cette prédiction
        if context_key in self._models_cache:
            self.logger.info(f"Réutilisation de prédiction en cache pour {context_key}")
            return self._models_cache[context_key]
        
        # Essayer de charger ou mettre à jour le modèle si nécessaire
        if self.transfer_model is None:
            # Si la KB existe mais pas le modèle, le charger
            if len(self.knowledge_base or []) >= 3:
                if not self._load_transfer_model():
                    # Si le chargement échoue, essayer de reconstruire
                    self._update_transfer_model()
        
        if self.transfer_model is None or not self.transfer_model:
            self.logger.warning("Pas de modèle de transfert disponible")
            return {param: 0.0 for param in parameters}
            
        try:
            # Extraire les caractéristiques du contexte
            context = self._extract_context_features(
                year=year,
                scenario=scenario,
                use_stochastic_schedules=use_stochastic_schedules
            )
            
            # Convertir en vecteur de caractéristiques
            features = np.array(list(context.values())).reshape(1, -1)
            
            # Prédire pour chaque paramètre demandé
            predictions = {}
            
            for param_name in parameters:
                # Vérifier si un modèle existe pour ce paramètre
                if param_name in self.transfer_model:
                    model_info = self.transfer_model[param_name]
                    model = model_info['model']
                    scaler_X = model_info['scaler_X']
                    scaler_y = model_info['scaler_y']
                    
                    # Standardiser les caractéristiques
                    features_scaled = scaler_X.transform(features)
                    
                    # Prédire
                    pred_scaled = model.predict(features_scaled)
                    
                    # Destandardiser la prédiction
                    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
                    
                    # Limiter aux bornes du paramètre
                    bounds = parameters[param_name]['bounds']
                    pred = max(bounds[0], min(bounds[1], pred))
                    
                    predictions[param_name] = pred
                else:
                    # Si pas de modèle, utiliser 0.0 (pas d'ajustement)
                    predictions[param_name] = 0.0
                    
            self.logger.info(f"Paramètres prédits pour {year}, {scenario}: {predictions}")

            # Stocker la prédiction en cache
            self._models_cache[context_key] = predictions
            return predictions
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction des paramètres: {str(e)}")
            return {param: 0.0 for param in parameters}
    
    def add_calibration_result(self, 
                             optimal_params: Dict[str, float],
                             metrics: Dict[str, float],
                             year: int,
                             scenario: str = 'baseline',
                             use_stochastic_schedules: bool = False,
                             schedule_stats: Optional[Dict] = None) -> None:
        """
        Ajoute un résultat de calibration à la base de connaissances.
        
        Args:
            optimal_params: Paramètres optimaux trouvés
            metrics: Métriques de performance
            year: Année de calibration
            scenario: Scénario de calibration
            use_stochastic_schedules: Utilisation d'horaires stochastiques
            schedule_stats: Statistiques des schedules utilisés (optionnel)
        """
        try:
            # S'assurer que la connaissance base est chargée (avec verrou)
            self._load_knowledge_base()
            
            # Extraire les caractéristiques du contexte
            context = self._extract_context_features(
                year=year,
                scenario=scenario,
                use_stochastic_schedules=use_stochastic_schedules
            )
            
            # Créer l'entrée de la base de connaissances
            entry = {
                'id': len(self.knowledge_base) + 1,
                'context': context,
                'optimal_params': optimal_params,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Si des statistiques de schedules sont fournies, les stocker
            if schedule_stats and use_stochastic_schedules:
                # Ajouter aux informations de contexte
                entry['schedule_stats'] = schedule_stats
                
                # Mettre à jour les statistiques globales par année
                if not hasattr(self, 'schedule_statistics'):
                    self.schedule_statistics = {}
                
                if year not in self.schedule_statistics:
                    self.schedule_statistics[year] = {}
                    
                # Fusionner avec les statistiques existantes
                for key, value in schedule_stats.items():
                    if key not in self.schedule_statistics[year]:
                        self.schedule_statistics[year][key] = value
                    else:
                        # Faire une moyenne pondérée si la statistique existe déjà
                        existing = self.schedule_statistics[year][key]
                        n_entries = len([e for e in self.knowledge_base 
                                        if e.get('context', {}).get('year') == year])
                        
                        # Pondération: (n * existing + new) / (n + 1)
                        self.schedule_statistics[year][key] = (n_entries * existing + value) / (n_entries + 1)
            
            # Ajouter à la base de connaissances avec verrou
            self.knowledge_base.append(entry)
            
            # Sauvegarder la base de connaissances (avec verrou)
            self._save_knowledge_base()
            
            # Mettre à jour le modèle de transfert si suffisamment de données
            if len(self.knowledge_base) >= 3:
                self._update_transfer_model()
                
            self.logger.info(f"Résultat de calibration ajouté à la base de connaissances (ID: {entry['id']})")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout du résultat de calibration: {str(e)}")
    
    def discover_previous_calibrations(self) -> int:
        """
        Découvre automatiquement les calibrations précédentes.
        
        Returns:
            Nombre de calibrations découvertes
        """
        try:
            # S'assurer que la KB est chargée
            self._load_knowledge_base()
            
            # Chercher tous les résultats de calibration
            calibration_dirs = []
            for year in range(2019, 2025):  # Années possibles
                year_dir = config.paths['output'].CALIBRATION / str(year)
                if year_dir.exists():
                    # Chercher les répertoires de campagne
                    for campaign_dir in year_dir.iterdir():
                        if campaign_dir.is_dir():
                            # Vérifier s'il s'agit d'une calibration complète
                            results_file = campaign_dir / 'calibration_results.json'
                            if results_file.exists():
                                calibration_dirs.append((year, campaign_dir))
            
            # Nombres de calibrations déjà dans la base
            existing_ids = {entry.get('campaign_id', '') for entry in self.knowledge_base if 'campaign_id' in entry}
            
            # Pour chaque calibration découverte
            new_entries = 0
            for year, campaign_dir in calibration_dirs:
                # Vérifier si déjà dans la base
                campaign_id = campaign_dir.name
                if campaign_id in existing_ids:
                    continue
                
                # Charger les résultats
                try:
                    results_file = campaign_dir / 'calibration_results.json'
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    # Extraire les informations nécessaires
                    optimal_params = results.get('best_params', {})
                    metrics = results.get('best_metrics', {})
                    
                    # Déterminer le scénario (baseline par défaut)
                    scenario = 'baseline'  # Par défaut
                    use_stochastic = results.get('stochastic', False)
                    
                    # Ajouter à la base de connaissances
                    self.add_calibration_result(
                        optimal_params=optimal_params,
                        metrics=metrics,
                        year=year,
                        scenario=scenario,
                        use_stochastic_schedules=use_stochastic
                    )
                    
                    # Enregistrer l'ID de campagne
                    if self.knowledge_base and len(self.knowledge_base) > 0:
                        self.knowledge_base[-1]['campaign_id'] = campaign_id
                        # Sauvegarder après chaque ajout pour éviter la perte en cas d'erreur
                        self._save_knowledge_base()
                    
                    new_entries += 1
                    self.logger.info(f"Calibration découverte: {year}, {campaign_id}")
                    
                except Exception as e:
                    self.logger.error(f"Erreur lors du traitement de {campaign_dir}: {str(e)}")
            
            # Sauvegarder la base de connaissances mise à jour (déjà fait après chaque ajout)
            
            # Mettre à jour le modèle de transfert si suffisamment de données et nouvelles entrées
            if len(self.knowledge_base) >= 3 and new_entries > 0:
                self._update_transfer_model()
            
            self.logger.info(f"{new_entries} nouvelles calibrations découvertes et ajoutées")
            return new_entries
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la découverte des calibrations: {str(e)}")
            return 0
    
    def analyze_knowledge_base(self, output_dir: Optional[Path] = None) -> Dict:
        """
        Analyse la base de connaissances pour extraire des insights.
        
        Args:
            output_dir: Répertoire de sortie pour les résultats
            
        Returns:
            Dictionnaire avec les résultats d'analyse
        """
        # S'assurer que la KB est chargée
        self._load_knowledge_base()
        
        if not self.knowledge_base:
            self.logger.warning("Base de connaissances vide, rien à analyser")
            return {}
            
        try:
            # Préparer le répertoire de sortie
            if output_dir is None:
                output_dir = ensure_dir(self.db_path / 'analysis')
            else:
                output_dir = ensure_dir(output_dir)
            
            # Extraire les données pour l'analyse
            entries = []
            params_dict = {}
            years = []
            
            for entry in self.knowledge_base:
                # Ajouter l'entrée formatée
                entry_dict = {
                    'id': entry['id'],
                    'year': entry['context']['year'],
                    'is_stochastic': entry['context']['is_stochastic'],
                    'rmse': entry['metrics'].get('rmse', float('nan')),
                    'timestamp': entry['timestamp'],
                }
                
                # Ajouter les paramètres
                for param, value in entry['optimal_params'].items():
                    entry_dict[f"param_{param}"] = value
                    
                    # Collecter les valeurs pour chaque paramètre
                    if param not in params_dict:
                        params_dict[param] = []
                    params_dict[param].append(value)
                
                entries.append(entry_dict)
                years.append(entry['context']['year'])
            
            # Convertir en DataFrame pour l'analyse
            df = pd.DataFrame(entries)
            
            # Analyser la distribution des paramètres optimaux
            param_stats = {}
            for param, values in params_dict.items():
                param_stats[param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            
            # Analyser l'évolution des performances
            performance_by_year = df.groupby('year')['rmse'].mean().to_dict()
            
            # Analyser les corrélations entre paramètres et performances
            correlations = {}
            param_cols = [col for col in df.columns if col.startswith('param_')]
            if len(df) > 3 and param_cols:  # Assez de données pour les corrélations
                corr_matrix = df[param_cols + ['rmse']].corr()
                for param in param_cols:
                    correlations[param.replace('param_', '')] = corr_matrix.loc[param, 'rmse']
            
            # Compiler les résultats
            analysis_results = {
                'n_entries': len(self.knowledge_base),
                'years_covered': sorted(set(years)),
                'parameter_stats': param_stats,
                'performance_by_year': performance_by_year,
                'parameter_correlations': correlations,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Sauvegarder les résultats
            save_results(analysis_results, output_dir / 'knowledge_base_analysis.json', 'json')
            
            # Générer un rapport explicatif
            report = self._generate_analysis_report(analysis_results)
            with open(output_dir / 'knowledge_base_analysis.md', 'w') as f:
                f.write(report)
                
            self.logger.info(f"Analyse de la base de connaissances terminée avec {len(df)} entrées")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de la base de connaissances: {str(e)}")
            return {}
    
    def _generate_analysis_report(self, analysis_results: Dict) -> str:
        """Génère un rapport explicatif à partir des résultats d'analyse."""
        report = "# Analyse de la Base de Connaissances UBEM\n\n"
        report += f"*Générée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*\n\n"
        
        report += f"## Aperçu général\n\n"
        report += f"- **Nombre d'entrées**: {analysis_results['n_entries']}\n"
        report += f"- **Années couvertes**: {', '.join(map(str, analysis_results['years_covered']))}\n\n"
        
        report += f"## Distribution des paramètres optimaux\n\n"
        report += "| Paramètre | Moyenne | Écart-type | Min | Max |\n"
        report += "|-----------|---------|------------|-----|-----|\n"
        
        for param, stats in analysis_results['parameter_stats'].items():
            report += f"| {param} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n"
        
        report += "\n## Évolution des performances\n\n"
        report += "| Année | RMSE moyen |\n"
        report += "|-------|------------|\n"
        
        for year, rmse in sorted(analysis_results['performance_by_year'].items()):
            report += f"| {year} | {rmse:.4f} |\n"
        
        if analysis_results['parameter_correlations']:
            report += "\n## Corrélations paramètres-performance\n\n"
            report += "| Paramètre | Corrélation avec RMSE |\n"
            report += "|-----------|----------------------|\n"
            
            sorted_corrs = sorted(
                analysis_results['parameter_correlations'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for param, corr in sorted_corrs:
                report += f"| {param} | {corr:.4f} |\n"
                
            report += "\n*Note: Une corrélation négative signifie qu'augmenter ce paramètre tend à réduire l'erreur (RMSE).*\n"
        
        report += "\n## Conclusions et recommandations\n\n"
        
        # Identifer les paramètres les plus influents
        if analysis_results['parameter_correlations']:
            top_params = sorted(
                analysis_results['parameter_correlations'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            
            report += "### Paramètres les plus influents\n\n"
            for param, corr in top_params:
                direction = "diminuer" if corr > 0 else "augmenter"
                report += f"- **{param}**: {direction} ce paramètre tend à améliorer les performances (corrélation: {corr:.4f})\n"
        
        # Identifier les tendances temporelles
        if len(analysis_results['performance_by_year']) > 1:
            years = sorted(analysis_results['performance_by_year'].keys())
            rmses = [analysis_results['performance_by_year'][y] for y in years]
            
            if years[-1] - years[0] > 1:  # Au moins 2 années d'écart
                if rmses[-1] < rmses[0]:
                    report += "\n### Tendance d'amélioration\n\n"
                    report += f"Les performances se sont améliorées de {rmses[0]:.4f} à {rmses[-1]:.4f} "
                    report += f"entre {years[0]} et {years[-1]}, soit une réduction de {(1 - rmses[-1]/rmses[0])*100:.1f}% de l'erreur.\n"
                else:
                    report += "\n### Tendance de dégradation\n\n"
                    report += f"Les performances se sont dégradées de {rmses[0]:.4f} à {rmses[-1]:.4f} "
                    report += f"entre {years[0]} et {years[-1]}, soit une augmentation de {(rmses[-1]/rmses[0] - 1)*100:.1f}% de l'erreur.\n"
                    report += "Cela peut indiquer des changements structurels dans les données ou le modèle qui nécessitent attention.\n"
        
        # Ajouter une section sur les schedules si des données existent
        if hasattr(self, 'schedule_statistics') and self.schedule_statistics:
            report += "\n## Analyse des Schedules\n\n"
            
            if len(self.schedule_statistics) > 1:
                # Comparer les années
                report += "### Comparaison par année\n\n"
                report += "| Année | Occupation moyenne | Ratio pic soirée | Pic matin |\n"
                report += "|-------|-------------------|------------------|----------|\n"
                
                for year, stats in sorted(self.schedule_statistics.items()):
                    report += f"| {year} | {stats.get('occupants_avg', 'N/A'):.2f} | "
                    report += f"{stats.get('evening_peak_ratio', 'N/A'):.2f} | "
                    report += f"{stats.get('morning_peak_ratio', 'N/A'):.2f} |\n"
                
                # Insight sur les tendances
                report += "\nL'analyse des schedules montre "
                
                # Exemple d'insight
                trend = "une stabilité dans les profils d'occupation"
                if len(self.schedule_statistics) >= 2:
                    years = sorted(self.schedule_statistics.keys())
                    first_occ = self.schedule_statistics[years[0]].get('occupants_avg', 0.5)
                    last_occ = self.schedule_statistics[years[-1]].get('occupants_avg', 0.5)
                    
                    if abs(last_occ - first_occ) > 0.1:
                        trend = f"une évolution des profils d'occupation de {first_occ:.2f} à {last_occ:.2f}"
                
                report += f"{trend} au fil des années.\n"
        
        return report
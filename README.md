# DocumentDetectorPOC

## Objectif principal
Le but de ce projet est de détecter les bords d'un document dans une image et de l'isoler. Cela implique :

1. Détection des contours : Utilisation de YOLOv5 pour localiser les documents dans des images.
2. Isolement : Extraction du document détecté afin de l'utiliser dans des contextes variés.
3. Conversion pour TensorFlow.js : Transformer un modèle .h5 en un format compatible avec TensorFlow.js pour une intégration dans l'interface React.

## Organisation du projet

```
DocumentDetectorPOC/
├── dataset/                    # Données pour l'entraînement
│   ├── generated/              # Données augmentées et labels
│   ├── original/               # Données brutes (images originales)
│   └── dataset.yaml            # Configuration des classes du dataset
├── document-detector-frontend/ # Interface React pour visualiser les résultats
│   └── public/js_model/        # Modèle TensorFlow.js converti
├── scripts/                    # Scripts pour le traitement des données et modèles
│   ├── generate_dataset.py     # Génération des données augmentées
│   ├── train_model.py          # Entraînement du modèle YOLOv5
│   └── convert_to_tfjs.py      # Conversion du modèle en TensorFlow.js
├── yolo_env/                   # Environnement virtuel Python pour l'entraînement
├── yolov5/                     # Code source de YOLOv5
├── train_yolov5.py             # Script d'entraînement principal
└── requirements.txt            # Dépendances Python
```

## Gestion des environnements Python

### Qu'est-ce qu'un environnement virtuel ?
Un environnement virtuel Python est un espace isolé où les dépendances nécessaires au projet sont installées, évitant les conflits avec les autres projets ou avec les paquets globaux installés sur la machine. Cela garantit la reproductibilité du projet et facilite son déploiement ou son partage.
Cela permet aussi d'avoir une version spécifique de Python nécessaire pour pourvoir faire tourner l'ensemble du projet sans avoir à modifier le version de Python utilisée sur la machine.

## Les environnements dans ce projet
Dans ce projet, deux environnements virtuels sont utilisés pour organiser et gérer les dépendances :

venv/ : Principalement pour les scripts Python généraux.
yolo_env/ : Spécifiquement pour YOLOv5 et ses dépendances, comme torch.

## Commandes pour gérer les environnements
Par contre attention les commandes suivantes ne fonctionnent que sur Windows

1. Activer l'environnement venv :

```bash
.\venv\Scripts\Activate
```

2. Activer l'environnement yolo_env :

```bash
.\yolo_env\Scripts\Activate
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

4. Désactiver un environnement :
Pour quitter l'environnement actif :

```bash
deactivate
```
## Pourquoi séparer les environnements ?
- Isolation des dépendances : Les paquets spécifiques à YOLO (comme torch ou ultralytics) peuvent être installés dans yolo_env sans affecter les autres parties du projet.
- Clarté et organisation : Les scripts généraux (par exemple, convert_to_tfjs.py) utilisent venv, tandis que les tâches liées à YOLOv5 utilisent yolo_env.
- Flexibilité : Cette séparation facilite la gestion et le dépannage, notamment si YOLO nécessite des versions spécifiques de bibliothèques comme torch.


## Étapes

1. Entraînement du modèle
Les images dans dataset/original/ sont transformées et labellisées automatiquement dans dataset/generated/.
Utiliser train_yolov5.py pour entraîner le modèle YOLOv5 

```bash
python train_yolov5.py
```
Le modèle entraîné est sauvegardé sous forme de fichier .pt

2. Conversion en fichier Keras (.h5)
Convertir le modèle YOLOv5 .pt en un modèle compatible avec Keras (.h5), en modifiant si nécessaire le script dans scripts/train_model.py.

3. Conversion pour TensorFlow.js
Une fois le modèle converti en .h5, utiliser le script suivant pour le convertir en modèle TensorFlow.js :

```bash
import tensorflowjs as tfjs
import tensorflow as tf

# Charger le modèle Keras
model = tf.keras.models.load_model('document_detector_model.h5', compile=False)

# Sauvegarder le modèle converti dans le frontend
tfjs.converters.save_keras_model(model, 'document-detector-frontend/public/js_model')
```

4. Utilisation dans React avec TensorFlow.js

Charger le modèle dans l'interface React et l'utiliser pour détecter les documents : 

```bash
import * as tf from '@tensorflow/tfjs';

const loadModel = async () => {
    const model = await tf.loadGraphModel('/js_model/model.json');
    console.log("Modèle chargé :", model);
};

loadModel();
```

## Commandes utiles

Entraînement du modèle :
```bash
python train_yolov5.py
```

Conversion en TensorFlow.js :
```bash
python convert_to_tfjs.py
```

Lancement du serveur React :
```bash
cd document-detector-frontend/
npm run dev
```

## Fonctionnalités principales

Détection des documents : Localiser les contours des documents avec YOLOv5.
Isolement des documents : Recadrage des images autour des bords détectés.
Intégration web : Déploiement du modèle avec TensorFlow.js et React.
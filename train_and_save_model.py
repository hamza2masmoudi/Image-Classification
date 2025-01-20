import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris  # Exemple de dataset

# Charger un dataset d'exemple
data = load_iris()
X, y = data.data, data.target

# Prétraitement des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraîner un modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Créer le dossier artifacts s'il n'existe pas
artifacts_path = './serving/artifacts'
os.makedirs(artifacts_path, exist_ok=True)

# Sauvegarder le modèle et le scaler
with open(os.path.join(artifacts_path, 'model.pkl'), 'wb') as model_file:
    pickle.dump(model, model_file)

with open(os.path.join(artifacts_path, 'scaler.pkl'), 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Modèle et scaler sauvegardés dans le dossier 'artifacts'.")
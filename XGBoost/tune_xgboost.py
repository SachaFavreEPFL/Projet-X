import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from itertools import product
import time
import matplotlib.pyplot as plt

# Charger les données d'entraînement
train_df = pd.read_csv('dataset/train_stocks_valuation_light.csv')

# Définir les features et la cible
features = [
    'pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio',
    'roe', 'roa', 'operating_margins', 'profit_margins',
    'debt_to_equity', 'current_ratio',
    'earnings_growth', 'revenue_growth',
    'beta', 'market_cap',
    'sector_encoded'
]
target = 'valuation_class_encoded'

X = train_df[features]
y = train_df[target]

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_config(config, X_train, X_test, y_train, y_test):
    model = XGBClassifier(
        **config,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Étape 1: Paramètres principaux
print("Étape 1: Recherche des paramètres principaux...")
param_grid_step1 = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

keys, values = zip(*param_grid_step1.items())
configs_to_test = [dict(zip(keys, v)) for v in product(*values)]

best_score = 0
best_config = None

for config in configs_to_test:
    print(f"Test de la configuration : {config}")
    score = evaluate_config(config, X_train, X_test, y_train, y_test)
    print(f"Score de précision : {score:.4f}\n")
    if score > best_score:
        best_score = score
        best_config = config
    time.sleep(0.2)

print(f"\nMeilleure configuration de l'étape 1 : {best_config}")
print(f"Score : {best_score:.4f}")

# Étape 2: Paramètres secondaires avec les meilleures valeurs de l'étape 1
print("\nÉtape 2: Optimisation fine des paramètres secondaires...")
param_grid_step2 = {
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
    "reg_alpha": [0, 0.1, 0.2],
    "reg_lambda": [0, 0.1, 0.2]
}

# Ajouter les meilleures valeurs de l'étape 1
for key, value in best_config.items():
    param_grid_step2[key] = [value]

keys, values = zip(*param_grid_step2.items())
configs_to_test = [dict(zip(keys, v)) for v in product(*values)]

for config in configs_to_test:
    print(f"Test de la configuration : {config}")
    score = evaluate_config(config, X_train, X_test, y_train, y_test)
    print(f"Score de précision : {score:.4f}\n")
    if score > best_score:
        best_score = score
        best_config = config
    time.sleep(0.2)

# Après la boucle d'optimisation, ajouter :
print("\nAnalyse de l'importance des features :")
best_model = XGBClassifier(
    **best_config,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
best_model.fit(X_train, y_train)

# Obtenir l'importance des features
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nImportance des features :")
print(feature_importance)

# Sauvegarde des résultats
result = {
    "best_config": best_config,
    "best_score": best_score,
    "feature_importance": feature_importance.to_dict('records')
}

OUTPUT_DIR = 'XGBoost'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(OUTPUT_DIR, 'xgb_best_config.json')

with open(CONFIG_PATH, 'w') as f:
    json.dump(result, f, indent=4)

print(f"\nMeilleure configuration finale sauvegardée dans {CONFIG_PATH} :")
print(json.dumps(result, indent=4, ensure_ascii=False))

# Créer un graphique d'importance des features
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Importance des Features')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
print("\nGraphique d'importance des features sauvegardé dans 'XGBoost/feature_importance.png'") 
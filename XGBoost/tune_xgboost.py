import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

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

# Liste des configurations à tester (modifiable)
configs_to_test = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.07, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 0.8}
]

best_score = 0
best_config = None

for config in configs_to_test:
    print(f"Test de la configuration : {config}")
    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Score de précision : {score:.4f}\n")
    if score > best_score:
        best_score = score
        best_config = config

# Sauvegarde de la meilleure configuration
result = {
    "best_config": best_config,
    "best_score": best_score
}

OUTPUT_DIR = 'XGBoost'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(OUTPUT_DIR, 'xgb_best_config.json')

with open(CONFIG_PATH, 'w') as f:
    json.dump(result, f, indent=4)

print(f"Meilleure configuration sauvegardée dans {CONFIG_PATH} :")
print(json.dumps(result, indent=4, ensure_ascii=False)) 
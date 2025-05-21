import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Chemins des fichiers
MODEL_DIR = 'XGBoost'
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.joblib')
DATA_PATH = 'dataset/train_stocks_valuation.csv'

def load_model():
    """
    Charge le modèle XGBoost entraîné
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Le modèle n'a pas été trouvé dans {MODEL_PATH}. Veuillez d'abord entraîner le modèle.")
    
    return joblib.load(MODEL_PATH)

def prepare_data():
    """
    Prépare les données pour la prédiction
    """
    # Chargement des données
    df = pd.read_csv(DATA_PATH)
    
    # Sélection des features
    features = [
        'pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio',
        'roe', 'roa', 'operating_margins', 'profit_margins',
        'debt_to_equity', 'current_ratio',
        'earnings_growth', 'revenue_growth',
        'beta', 'market_cap',
        'sector_encoded'
    ]
    
    # Vérification des valeurs manquantes
    print("\nVérification des valeurs manquantes :")
    print(df[features].isnull().sum())
    
    # Vérification des statistiques descriptives
    print("\nStatistiques des features :")
    print(df[features].describe())
    
    # Vérification de la distribution des classes (si disponible)
    if 'valuation_class_encoded' in df.columns:
        print("\nDistribution des classes :")
        print(df['valuation_class_encoded'].value_counts(normalize=True).round(3) * 100)
    
    return df[features], df

def main():
    try:
        print("Chargement du modèle...")
        model = load_model()
        
        print("\nChargement des données...")
        X, df = prepare_data()
        
        # Vérification colonne symbol
        print("\nVérification de la colonne 'symbol' :")
        n_missing = df['symbol'].isnull().sum()
        n_duplicates = df['symbol'].duplicated().sum()
        print(f"Valeurs manquantes : {n_missing}")
        print(f"Doublons : {n_duplicates}")
        if n_missing > 0:
            print("[ALERTE] Il y a des valeurs manquantes dans la colonne 'symbol'.")
        if n_duplicates > 0:
            print("[ALERTE] Il y a des doublons dans la colonne 'symbol'.")
        
        print("\nFaisant les prédictions...")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Créer un dictionnaire de mapping symbol -> prédiction
        prediction_dict = dict(zip(df['symbol'], predictions))
        confidence_dict = dict(zip(df['symbol'], np.max(probabilities, axis=1)))
        
        # Ajouter les colonnes de prédiction
        df['xgboost_prediction'] = df['symbol'].map(prediction_dict)
        df['xgboost_confidence'] = df['symbol'].map(confidence_dict)
        
        # Vérification de l'ajout des colonnes
        print("\nColonnes du DataFrame avant sauvegarde :")
        print(df.columns)
        print("Exemple de lignes :")
        print(df[['symbol','xgboost_prediction','xgboost_confidence']].head())
        
        # Sauvegarder directement dans le fichier source
        df.to_csv(DATA_PATH, index=False)
        
        # Afficher uniquement la précision et la confiance
        if 'valuation_class_encoded' in df.columns:
            accuracy = accuracy_score(df['valuation_class_encoded'], predictions)
            print(f"\nPrécision : {accuracy:.2%}")
        print(f"Confiance moyenne : {np.mean(probabilities.max(axis=1)):.2%}")
        print(f"\nFichier mis à jour : {DATA_PATH}")
        
    except Exception as e:
        print(f"\nErreur : {str(e)}")

if __name__ == "__main__":
    main()
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Chemins des fichiers
MODEL_DIR = 'XGBoost'
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.joblib')
CONFIG_PATH = os.path.join(MODEL_DIR, 'xgb_best_config.json')
DATA_PATH = 'dataset/stocks_extracted_filtered.csv'

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
    
    # Features utilisées dans tune_xgboost.py
    features = [
        # Ratios de valorisation
        'pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio', 'enterprise_to_ebitda',
        
        # Ratios de qualité
        'return_on_equity', 'return_on_assets', 'operating_margins', 'profit_margins',
        'debt_to_equity', 'current_ratio', 'quick_ratio', 'interest_coverage',
        
        # Ratios de croissance
        'earnings_growth', 'revenue_growth',
        
        # Ratios de marché
        'beta', 'market_daily_change',
        
        # Ratios financiers
        'bs_total_liabilities', 'bs_total_equity', 'bs_cash', 'bs_short_term_investments'
    ]
    
    # Filtrer les colonnes qui existent dans le DataFrame
    features = [f for f in features if f in df.columns]
    
    # Vérification des valeurs manquantes
    print("\nVérification des valeurs manquantes :")
    print(df[features].isnull().sum())
    
    # Vérification des statistiques descriptives
    print("\nStatistiques des features :")
    print(df[features].describe())
    
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
        
        # Créer un dictionnaire de mapping symbol -> prédiction
        prediction_dict = dict(zip(df['symbol'], predictions))
        
        # Ajouter les colonnes de prédiction
        df['xgboost_prediction'] = df['symbol'].map(prediction_dict)
        
        # Vérification de l'ajout des colonnes
        print("\nColonnes du DataFrame avant sauvegarde :")
        print(df.columns)
        print("Exemple de lignes :")
        print(df[['symbol', 'xgboost_prediction']].head())
        
        # Sauvegarder les résultats dans le fichier source
        df.to_csv(DATA_PATH, index=False)
        
        # Calculer les métriques si la vraie valeur est disponible
        if 'overvaluation_score' in df.columns:
            rmse = np.sqrt(mean_squared_error(df['overvaluation_score'], predictions))
            r2 = r2_score(df['overvaluation_score'], predictions)
            print(f"\nMétriques de performance :")
            print(f"RMSE : {rmse:.4f}")
            print(f"R² : {r2:.4f}")
        
        print(f"\nFichier mis à jour : {DATA_PATH}")
        
    except Exception as e:
        print(f"\nErreur : {str(e)}")

if __name__ == "__main__":
    main()
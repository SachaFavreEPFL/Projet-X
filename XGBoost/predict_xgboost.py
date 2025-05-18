import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

# Chemins des fichiers
MODEL_DIR = 'XGBoost'
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.joblib')
LIGHT_DATA_PATH = 'dataset/train_stocks_valuation_light.csv'
FULL_DATA_PATH = 'dataset/train_stocks_valuation.csv'

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
    df = pd.read_csv(LIGHT_DATA_PATH)
    
    # Sélection des features
    features = [
        'pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio',
        'roe', 'roa', 'operating_margins', 'profit_margins',
        'debt_to_equity', 'current_ratio',
        'earnings_growth', 'revenue_growth',
        'beta', 'market_cap',
        'sector_encoded'
    ]
    
    return df[features], df

def main():
    try:
        print("Chargement du modèle...")
        model = load_model()
        
        print("\nChargement des données...")
        X, light_df = prepare_data()
        
        print("\nFaisant les prédictions...")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Charger le fichier complet
        print("\nAjout des prédictions au fichier complet...")
        full_df = pd.read_csv(FULL_DATA_PATH)
        
        # Créer un dictionnaire de mapping symbol -> prédiction
        prediction_dict = dict(zip(light_df['symbol'], predictions))
        confidence_dict = dict(zip(light_df['symbol'], np.max(probabilities, axis=1)))
        
        # Ajouter les colonnes de prédiction
        full_df['xgboost_prediction'] = full_df['symbol'].map(prediction_dict)
        full_df['xgboost_confidence'] = full_df['symbol'].map(confidence_dict)
        
        # Sauvegarder le fichier mis à jour
        output_path = 'dataset/train_stocks_valuation_with_predictions.csv'
        full_df.to_csv(output_path, index=False)
        
        print(f"\nTraitement terminé !")
        print(f"Fichier mis à jour sauvegardé dans : {output_path}")
        
        # Afficher quelques statistiques
        print("\nStatistiques des prédictions :")
        print(f"Nombre total de prédictions : {len(predictions)}")
        print("\nDistribution des classes prédites :")
        print(pd.Series(predictions).value_counts(normalize=True).round(3) * 100)
        print("\nConfiance moyenne des prédictions : {:.2f}%".format(np.mean(probabilities.max(axis=1)) * 100))
        
    except Exception as e:
        print(f"\nErreur : {str(e)}")

if __name__ == "__main__":
    main()
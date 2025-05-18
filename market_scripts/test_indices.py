import pandas as pd
import json
import os

def get_stock_components():
    """Récupère les composants du marché boursier via le fichier CSV"""
    try:
        print("\nAnalyse du marché boursier...")
        
        # Lire le fichier CSV
        df = pd.read_csv('c:/Users/sacha/Downloads/listing_status.csv')
        
        # Filtrer pour ne garder que les actions actives
        active_stocks = df[
            (df['status'] == 'Active') & 
            (df['assetType'] == 'Stock')
        ]
        
        # Extraire les symboles
        symbols = active_stocks['symbol'].tolist()
        
        print(f"Symboles trouvés : {len(symbols)}")
        print(f"Exemple : {symbols[:5]}")
        
        return symbols
            
    except Exception as e:
        print(f"❌ Erreur : {str(e)}")
        return []

def save_symbols_to_file(symbols):
    """Sauvegarde les symboles dans le fichier stocks_to_analyze.json"""
    try:
        output_file = "dataset/stocks_to_analyze.json"
        
        # Créer le dossier dataset s'il n'existe pas
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Structure de base
        data = {
            "stocks": symbols
        }
        
        # Sauvegarder les données
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"✅ Symboles sauvegardés avec succès dans {output_file}")
            
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde des symboles: {str(e)}")

def main():
    print("Récupération des composants du marché boursier...")
    symbols = get_stock_components()
    
    if symbols:
        print("\nRÉSULTATS :")
        print("="*30)
        print(f"Nombre total d'actions : {len(symbols)}")
        print(f"Nombre d'actions uniques : {len(set(symbols))}")
        print(f"Nombre de doublons : {len(symbols) - len(set(symbols))}")
        
        # Sauvegarder les symboles
        save_symbols_to_file(symbols)

if __name__ == "__main__":
    main()
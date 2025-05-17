import json
from pathlib import Path

def reset_stocks_file():
    # Chemin du fichier JSON
    output_path = Path("dataset/stocks_to_analyze.json")
    
    # Créer la structure du fichier avec une liste vide
    stocks_data = {
        "stocks": []
    }
    
    # Sauvegarder dans le fichier JSON
    with open(output_path, 'w') as f:
        json.dump(stocks_data, f, indent=4)
    
    print(f"Liste des actions réinitialisée")
    print(f"Fichier mis à jour : {output_path}")

if __name__ == "__main__":
    reset_stocks_file() 
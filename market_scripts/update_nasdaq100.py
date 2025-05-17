import json
from pathlib import Path

def get_stocks_list():
    # Liste des actions du NASDAQ 100
    stocks = set()  # Utilisation d'un set pour éviter les doublons
    
    # NASDAQ 100 (liste complète)
    nasdaq100 = {
        "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA", "PEP", "COST",
        "ADBE", "CSCO", "NFLX", "INTC", "QCOM", "PYPL", "AMD", "SBUX", "AMAT", "ISRG",
        "ORCL", "INTU", "GILD", "MDLZ", "REGN", "KLAC", "KDP", "SNPS", "ASML", "CDNS",
        "ADP", "ADI", "ANSS", "BIIB", "CHTR", "CMCSA", "CTAS", "CTSH", "DLTR", "EA",
        "EBAY", "FAST", "IDXX", "ILMN", "INCY", "LRCX", "MAR", "MCHP", "MELI", "MNST",
        "MU", "NXPI", "PAYX", "PCAR", "ROST", "SIRI", "SWKS", "TMUS", "TXN", "VRTX",
        "WBA", "WDC", "WLTW", "WYNN", "XEL", "XLNX", "ZM", "AVGO", "BIDU", "BMRN",
        "CELG", "CERN", "CHKP", "CSX", "CTXS", "DXCM", "EXPE", "FISV", "FOX", "FOXA",
        "HAS", "HOLX", "HSIC", "JD", "KHC", "LBTYA", "LBTYK", "LULU", "MYL", "NTES",
        "ORLY", "SGEN", "ULTA", "VRSK", "WLTW"
    }
    stocks.update(nasdaq100)
    
    return sorted(list(stocks))

def update_stocks_file():
    # Chemin du fichier JSON
    output_path = Path("dataset/stocks_to_analyze.json")
    
    # Lire le fichier existant s'il existe
    existing_stocks = set()
    if output_path.exists():
        with open(output_path, 'r') as f:
            data = json.load(f)
            existing_stocks = set(data.get("stocks", []))
    
    # Récupérer la liste des actions du NASDAQ 100
    nasdaq_stocks = set(get_stocks_list())
    
    # Fusionner les deux ensembles
    all_stocks = existing_stocks.union(nasdaq_stocks)
    
    # Créer la structure du fichier
    stocks_data = {
        "stocks": sorted(list(all_stocks))
    }
    
    # Sauvegarder dans le fichier JSON
    with open(output_path, 'w') as f:
        json.dump(stocks_data, f, indent=4)
    
    print(f"Nombre total d'actions : {len(all_stocks)}")
    print(f"Fichier mis à jour : {output_path}")

if __name__ == "__main__":
    update_stocks_file() 
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Chargement des données
with open('dataset/stocks_data.json', 'r') as f:
    stocks_data = json.load(f)['stocks']
    print(f"\nNombre total d'actions dans stocks_data.json : {len(stocks_data)}")
with open('dataset/stocks_history.json', 'r') as f:
    stocks_history = json.load(f)['stocks']

def clean_value(value):
    """Nettoie une valeur en remplaçant les valeurs aberrantes par None"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if np.isinf(value) or np.isnan(value):
            return None
        # Limites raisonnables pour les ratios
        if value < -1000 or value > 1000:
            return None
    return value

def calculate_market_cap(stock_data):
    """Calcule le market_cap en utilisant le P/E ratio"""
    overview = stock_data.get('overview', {}).get('data', {})
    income_statement = stock_data.get('income_statement', {}).get('data', {})
    
    # Récupération des valeurs nécessaires
    pe_ratio = clean_value(overview.get('pe_ratio', None))
    net_income = clean_value(income_statement.get('net_income', None))
    
    # Debug logs
    print(f"\nDebug market_cap calculation:")
    print(f"pe_ratio: {pe_ratio}")
    print(f"net_income: {net_income}")
    
    # Si les deux valeurs sont disponibles, on calcule le market_cap
    if pe_ratio is not None and net_income is not None:
        market_cap = pe_ratio * net_income
        print(f"Calculated market_cap: {market_cap}")
        return market_cap
    print("Could not calculate market_cap - missing values")
    return None

def extract_features(symbol, stock):
    """Extraction des features pour chaque action"""
    overview = stock.get('overview', {}).get('data', {})
    balance_sheet = stock.get('balance_sheet', {}).get('data', {})
    cash_flow = stock.get('cash_flow', {}).get('data', {})
    market_data = stock.get('market_data', {}).get('data', {})
    income_statement = stock.get('income_statement', {}).get('data', {})
    
    features = {}
    
    # Données de base
    features['symbol'] = symbol
    features['sector'] = overview.get('sector', None)
    features['industry'] = overview.get('industry', None)
    
    # Overview features
    for key in overview:
        features[key] = clean_value(overview.get(key, None))
    
    # Balance sheet features
    for key in balance_sheet:
        features[f'bs_{key}'] = clean_value(balance_sheet.get(key, None))
    
    # Cash flow features
    for key in cash_flow:
        features[f'cf_{key}'] = clean_value(cash_flow.get(key, None))
    
    # Market data features
    for key in market_data:
        features[f'market_{key}'] = clean_value(market_data.get(key, None))
    
    # Income statement features
    for key in income_statement:
        features[f'is_{key}'] = clean_value(income_statement.get(key, None))
    
    return features

# Extraction des features
data = [extract_features(sym, stock) for sym, stock in stocks_data.items()]
df = pd.DataFrame(data)

print(f"\nNombre d'actions après extraction : {len(df)}")

# Calcul du pourcentage de valeurs non nulles pour chaque colonne
non_null_percentages = (df.count() / len(df) * 100)

# Filtrer les colonnes avec au moins 90% de valeurs non nulles
valid_columns = non_null_percentages[non_null_percentages >= 90].index
df_filtered = df[valid_columns]

print(f"\nNombre de colonnes avant filtrage : {len(df.columns)}")
print(f"Nombre de colonnes après filtrage : {len(df_filtered.columns)}")

# Analyse détaillée de chaque colonne restante
print("\nAnalyse détaillée des features conservées :")
for column in df_filtered.columns:
    if column not in ['symbol', 'sector', 'industry']:  # Skip non-numeric columns
        total = len(df_filtered)
        non_null = df_filtered[column].count()
        nulls = df_filtered[column].isnull().sum()
        
        # Conversion en numérique si possible
        try:
            numeric_series = pd.to_numeric(df_filtered[column], errors='coerce')
            zeros = (numeric_series == 0).sum()
            
            print(f"\n{column}:")
            print(f"  Total valeurs: {total}")
            print(f"  Valeurs non-nulles: {non_null} ({non_null/total*100:.1f}%)")
            print(f"  Valeurs nulles: {nulls} ({nulls/total*100:.1f}%)")
            print(f"  Valeurs = 0: {zeros} ({zeros/total*100:.1f}%)")
            
            if non_null > 0:
                print(f"  Min: {numeric_series.min()}")
                print(f"  Max: {numeric_series.max()}")
                print(f"  Moyenne: {numeric_series.mean():.2f}")
                print(f"  Médiane: {numeric_series.median():.2f}")
        except:
            print(f"\n{column}:")
            print(f"  Total valeurs: {total}")
            print(f"  Valeurs non-nulles: {non_null} ({non_null/total*100:.1f}%)")
            print(f"  Valeurs nulles: {nulls} ({nulls/total*100:.1f}%)")
            print("  Type: Non numérique")

# Sauvegarde du dataset filtré
df_filtered.to_csv('dataset/train_stocks_valuation.csv', index=False)
print("\nDataset généré : dataset/train_stocks_valuation.csv") 
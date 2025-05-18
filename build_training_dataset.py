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

def extract_features(symbol, stock):
    overview = stock.get('overview', {}).get('data', {})
    sector = overview.get('sector', None)
    
    # Ratios de valorisation
    pe = clean_value(overview.get('pe_ratio', None))
    pb = clean_value(overview.get('pb_ratio', None))
    ps = clean_value(overview.get('ps_ratio', None))
    earnings_growth = clean_value(overview.get('earnings_growth', None))
    
    # Calcul du PEG (éviter la division par zéro ou négatif)
    peg = None
    try:
        if pe is not None and earnings_growth is not None and earnings_growth > 0:
            peg = clean_value(pe / earnings_growth)
    except Exception:
        peg = None
    
    # Qualité
    roe = clean_value(overview.get('return_on_equity', None))
    roa = clean_value(overview.get('return_on_assets', None))
    operating_margins = clean_value(overview.get('operating_margins', None))
    profit_margins = clean_value(overview.get('profit_margins', None))
    debt_to_equity = clean_value(overview.get('debt_to_equity', None))
    current_ratio = clean_value(overview.get('current_ratio', None))
    
    # Croissance
    revenue_growth = clean_value(overview.get('revenue_growth', None))
    
    # Marché
    beta = clean_value(overview.get('beta', None))
    market_cap = clean_value(overview.get('market_cap', None))
    
    return {
        'symbol': symbol,
        'sector': sector,
        # Valorisation
        'pe_ratio': pe,
        'pb_ratio': pb,
        'ps_ratio': ps,
        'peg_ratio': peg,
        # Qualité
        'roe': roe,
        'roa': roa,
        'operating_margins': operating_margins,
        'profit_margins': profit_margins,
        'debt_to_equity': debt_to_equity,
        'current_ratio': current_ratio,
        # Croissance
        'earnings_growth': earnings_growth,
        'revenue_growth': revenue_growth,
        # Marché
        'beta': beta,
        'market_cap': market_cap
    }

# Extraction des features
data = [extract_features(sym, stock) for sym, stock in stocks_data.items()]
df = pd.DataFrame(data)

print(f"\nNombre d'actions après extraction initiale : {len(df)}")

# Analyse des valeurs manquantes par colonne
print("\nValeurs manquantes par colonne après extraction :")
missing_values = df.isnull().sum()
print(missing_values)

# Sauvegarde du nombre d'actions avant filtrage
df_before = len(df)

# Suppression des lignes sans données essentielles
essential_cols = ['sector', 'pe_ratio', 'pb_ratio', 'ps_ratio']  # Retrait du peg_ratio des colonnes essentielles
df = df.dropna(subset=essential_cols)

print(f"\nNombre d'actions après filtrage des données essentielles : {len(df)}")
print(f"Actions perdues : {df_before - len(df)}")

# Recalcul systématique du PEG
df['peg_ratio'] = df.apply(
    lambda row: row['pe_ratio'] / row['earnings_growth'] 
    if pd.notnull(row['pe_ratio']) and pd.notnull(row['earnings_growth']) and row['earnings_growth'] > 0 
    else None, 
    axis=1
)

# Nettoyage des valeurs extrêmes du PEG
df['peg_ratio'] = df['peg_ratio'].clip(lower=0, upper=10)  # Limitation des valeurs extrêmes

# Analyse de la distribution des PEG
print("\nAnalyse de la distribution des PEG :")
peg_stats = df['peg_ratio'].describe()
print("\nStatistiques des PEG :")
print(peg_stats)

# Analyse des PEG par secteur
print("\nMédiane des PEG par secteur :")
peg_by_sector = df.groupby('sector')['peg_ratio'].median().sort_values()
print(peg_by_sector)

# Identification des valeurs extrêmes
print("\nActions avec PEG extrêmes :")
extreme_peg = df[df['peg_ratio'].notna()].sort_values('peg_ratio')
print("\n5 actions avec les PEG les plus bas :")
print(extreme_peg[['symbol', 'sector', 'peg_ratio', 'pe_ratio', 'earnings_growth']].head())
print("\n5 actions avec les PEG les plus élevés :")
print(extreme_peg[['symbol', 'sector', 'peg_ratio', 'pe_ratio', 'earnings_growth']].tail())

# Analyse des valeurs manquantes par colonne essentielle
print("\nValeurs manquantes par colonne essentielle :")
missing_essential = df[essential_cols].isnull().sum()
print(missing_essential)

# Analyse des valeurs aberrantes
print("\nAnalyse des valeurs aberrantes par colonne :")
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'market_cap':
        outliers = df[(df[col] < -1000) | (df[col] > 1000)][col].count()
        if outliers > 0:
            print(f"{col}: {outliers} valeurs aberrantes")

# Remplacement des valeurs aberrantes restantes par la médiane du secteur
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'market_cap':  # Ne pas traiter la capitalisation boursière
        df[col] = df.groupby('sector')[col].transform(
            lambda x: x.fillna(x.median())
        )

# Définition des colonnes par catégorie
valuation_cols = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio']  # Utilisation du peg_ratio
quality_cols = ['roe', 'roa', 'operating_margins', 'profit_margins']
growth_cols = ['earnings_growth', 'revenue_growth']
market_cols = ['beta']

# Calcul des médianes par secteur pour les ratios de valorisation
sector_medians = df.groupby('sector')[valuation_cols].median()
sector_medians = sector_medians.rename(columns={col: f'{col}_sector_median' for col in valuation_cols})
df = df.merge(sector_medians, left_on='sector', right_index=True)

# Calcul des écarts en pourcentage
for col in valuation_cols:
    df[f'{col}_gap'] = (df[col] - df[f'{col}_sector_median']) / df[f'{col}_sector_median'] * 100

# Scores par catégorie
df['valuation_score'] = (
    0.3 * df['pe_ratio_gap'] +
    0.25 * df['pb_ratio_gap'] +
    0.25 * df['ps_ratio_gap'] +
    0.2 * df['peg_ratio_gap']
)

# Normalisation des scores de qualité (0-100)
for col in quality_cols:
    if col in df.columns:
        df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100

df['quality_score'] = (
    0.3 * df['roe_normalized'] +
    0.2 * df['roa_normalized'] +
    0.3 * df['operating_margins_normalized'] +
    0.2 * df['profit_margins_normalized']
)

# Normalisation des scores de croissance
for col in growth_cols:
    if col in df.columns:
        df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100

df['growth_score'] = (
    0.5 * df['earnings_growth_normalized'] +
    0.5 * df['revenue_growth_normalized']
)

# Score de marché (inverse du beta, normalisé)
if 'beta' in df.columns:
    df['market_score'] = (1 / df['beta']).fillna(0)
    df['market_score'] = (df['market_score'] - df['market_score'].min()) / (df['market_score'].max() - df['market_score'].min()) * 100
else:
    df['market_score'] = 50  # Valeur neutre si beta non disponible

# Score global pondéré
df['final_score'] = (
    0.4 * df['valuation_score'] +
    0.3 * df['quality_score'] +
    0.2 * df['growth_score'] +
    0.1 * df['market_score']
)

# Classification
def classify_valuation(score):
    if score <= -20:  # Ajusté de -30 à -20
        return 'Fortement sous-évaluée'
    elif score <= -5:  # Ajusté de -10 à -5
        return 'Légèrement sous-évaluée'
    elif score <= 5:   # Ajusté de 10 à 5
        return 'Normale'
    elif score <= 20:  # Ajusté de 30 à 20
        return 'Légèrement surévaluée'
    else:             # > 20
        return 'Fortement surévaluée'

df['valuation_class'] = df['final_score'].apply(classify_valuation)

# Calcul du score de solidité financière
# Normalisation des indicateurs de solidité
if 'debt_to_equity' in df.columns:
    df['debt_to_equity_normalized'] = (df['debt_to_equity'].max() - df['debt_to_equity']) / (df['debt_to_equity'].max() - df['debt_to_equity'].min()) * 100
else:
    df['debt_to_equity_normalized'] = 50

if 'revenue_growth' in df.columns:
    df['revenue_growth_normalized'] = (df['revenue_growth'] - df['revenue_growth'].min()) / (df['revenue_growth'].max() - df['revenue_growth'].min()) * 100
else:
    df['revenue_growth_normalized'] = 50

if 'current_ratio' in df.columns:
    df['current_ratio_normalized'] = (df['current_ratio'] - df['current_ratio'].min()) / (df['current_ratio'].max() - df['current_ratio'].min()) * 100
else:
    df['current_ratio_normalized'] = 50

# Score de solidité financière (moyenne pondérée)
df['financial_strength_score'] = (
    0.4 * df['debt_to_equity_normalized'] +
    0.3 * df['revenue_growth_normalized'] +
    0.3 * df['current_ratio_normalized']
)

# Sauvegarde du dataset
train_cols = [
    'symbol', 'sector',
    # Valorisation
    'pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio',
    # Qualité
    'roe', 'roa', 'operating_margins', 'profit_margins',
    'debt_to_equity', 'current_ratio',
    # Croissance
    'earnings_growth', 'revenue_growth',
    # Marché
    'beta', 'market_cap',
    # Scores
    'valuation_score', 'quality_score', 'growth_score', 'market_score',
    'final_score', 'valuation_class',
    # Solidité financière
    'financial_strength_score'
]

df[train_cols].to_csv('dataset/train_stocks_valuation.csv', index=False)

# Création du fichier léger avec les variables brutes
raw_cols = [
    'symbol',
    # Valorisation
    'pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio',
    # Qualité
    'roe', 'roa', 'operating_margins', 'profit_margins',
    'debt_to_equity', 'current_ratio',
    # Croissance
    'earnings_growth', 'revenue_growth',
    # Marché
    'beta', 'market_cap'
]

# Encodage du secteur
le_sector = LabelEncoder()
df['sector_encoded'] = le_sector.fit_transform(df['sector'])

# Encodage de la classe de valorisation
le_valuation = LabelEncoder()
df['valuation_class_encoded'] = le_valuation.fit_transform(df['valuation_class'])

# Sauvegarde du mapping des secteurs
sector_mapping = dict(zip(le_sector.classes_, le_sector.transform(le_sector.classes_)))
print("\nMapping des secteurs :")
for sector, code in sector_mapping.items():
    print(f"{code}: {sector}")

# Sauvegarde du mapping des classes de valorisation
valuation_mapping = dict(zip(le_valuation.classes_, le_valuation.transform(le_valuation.classes_)))
print("\nMapping des classes de valorisation :")
for valuation, code in valuation_mapping.items():
    print(f"{code}: {valuation}")

# Ajout des colonnes encodées
raw_cols.insert(1, 'sector_encoded')
raw_cols.append('valuation_class_encoded')

# Sauvegarde du dataset léger
df[raw_cols].to_csv('dataset/train_stocks_valuation_light.csv', index=False)

# Affichage des statistiques
print("\nDistribution des classes de valorisation :")
print(df['valuation_class'].value_counts(normalize=True).round(3) * 100)

print("\nCorrélations avec le score final :")
correlations = df[['valuation_score', 'quality_score', 'growth_score', 'market_score', 'final_score']].corr()
print(correlations['final_score'].sort_values(ascending=False))

print("\nDataset d'entraînement généré : dataset/train_stocks_valuation.csv") 
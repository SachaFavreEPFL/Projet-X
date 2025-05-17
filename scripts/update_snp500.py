import json
from pathlib import Path

def get_stocks_list():
    # Liste des actions du S&P 500
    stocks = set()  # Utilisation d'un set pour éviter les doublons
    
    # S&P 500 (liste complète)
    sp500 = {
        "A", "AAL", "AAP", "AAPL", "ABBV", "ABC", "ABMD", "ABT", "ACN", "ADBE",
        "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ",
        "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL", "ALLE", "AMAT", "AMCR",
        "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "ANTM",
        "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "ATVI", 
        "AVB", "AVGO", "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BAX",
        "BBWI", "BBY", "BDX", "BEN", "BF.B", "BIIB", "BIO", "BK", "BKNG",
        "BKR", "BLK", "BLL", "BMY", "BR", "BRK.B", "BRO", "BRX", "BSX",
        "BWA", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", 
        "CBRE", "CCI", "CCL", "CDAY", "CDW", "CE", "CEG", "CF", "CFG", 
        "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", 
        "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COST", 
        "CPB", "CPRT", "CRL", "CRM", "CSCO", "CSX", "CTAS", "CTLT", "CTSH", 
        "CTVA", "CTXS", "CVS", "CVX", "CZR", "D", "DAL", "DD", "DE", "DFS", 
        "DG", "DGX", "DHI", "DHR", "DIS", "DISH", "DLR", "DLTR", "DOV", "DOW", 
        "DPZ", "DRE", "DRI", "DTE", "DUK", "DVA", "DVN", "DXC", "DXCM", "EA", 
        "EBAY", "ECL", "ED", "EFX", "EIX", "EMN", "EMR", "ENPH", "EOG", "EPAM", 
        "EQR", "ES", "ESS", "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", 
        "EXPE", "EXR", "F", "FANG", "FAST", "FB", "FBIN", "FCX", "FDS", "FDX", 
        "FE", "FFIV", "FIS", "FISV", "FITB", "FLT", "FMC", "FOX", "FOXA", 
        "FRC", "FRT", "FTNT", "FTV", "GD", "GE", "GILD", "GIS", "GL", "GLW", 
        "GM", "GNRC", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", 
        "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", 
        "HRL", "HSIC", "HST", "HSY", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", 
        "IFF", "ILMN", "INCY", "INTC", "INTU", "IP", "IPG", "IPGP", "IQV", "IR", 
        "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JCI", "JKHY", "JNJ", 
        "JNPR", "JPM", "K", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", 
        "KMX", "KO", "KR", "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", 
        "LMT", "LNC", "LNT", "LOW", "LRCX", "LUMN", "LUV", "LVS", "LW", "LYB", 
        "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", 
        "MDT", "MET", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", 
        "MO", "MOS", "MPC", "MPWR", "MRK", "MRO", "MS", "MSCI", "MSFT", "MSI", 
        "MTB", "MTCH", "MTD", "MU", "MXIM", "NCLH", "NDAQ", "NDSN", "NDX", 
        "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", 
        "NTRS", "NUE", "NVDA", "NVR", "NWL", "NWS", "NWSA", "NXPI", "O", "ODFL", 
        "OGN", "OKE", "OMC", "ON", "ORCL", "OTIS", "OXY", "PAYC", "PAYX", "PCAR", 
        "PCG", "PEAK", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", 
        "PKI", "PLD", "PM", "PNC", "PNR", "PNW", "POOL", "PPG", "PPL", "PRU", "PSA", 
        "PSX", "PTC", "PWR", "PXD", "PYPL", "QCOM", "QRVO", "RE", "REG", "REGN", "RF", 
        "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "SBAC", "SBNY", 
        "SBSI", "SEDG", "SEE", "SHW", "SIVB", "SJM", "SLB", "SNA", "SNPS", "SO", "SPG", 
        "SPGI", "SRE", "STE", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", 
        "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", 
        "TMO", "TMUS", "TPR", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", 
        "TTWO", "TXN", "TXT", "TYL", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", 
        "UPS", "URI", "USB", "V", "VFC", "VICI", "VLO", "VMC", "VNO", "VRSK", 
        "VRSN", "VRTX", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WDC", "WEC", 
        "WELL", "WFC", "WHR", "WM", "WMB", "WRB", "WRK", "WST", "WTW", "WY", "WYNN", 
        "XEL", "XLNX", "XOM", "XRAY", "XYL", "YUM", "ZBRA", "ZBH", "ZION", "ZTS"
    }
    stocks.update(sp500)
    
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
    
    # Récupérer la liste des actions du S&P 500
    sp500_stocks = set(get_stocks_list())
    
    # Fusionner les deux ensembles
    all_stocks = existing_stocks.union(sp500_stocks)
    
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
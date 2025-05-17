import json
from pathlib import Path

def get_stocks_list():
    # Liste des actions du Russell 2000
    stocks = set()  # Utilisation d'un set pour éviter les doublons
    
    # Russell 2000 (liste étendue des actions)
    russell2000 = {
        # Technologie
        "PLTR", "RBLX", "COIN", "NET", "CRWD", "DDOG", "ZS", "OKTA", "TWLO", "DOCN",
        "FROG", "ESTC", "DBX", "PINS", "SNAP", "TWTR", "UBER", "LYFT", "ABNB", "DASH",
        "SHOP", "SQ", "PYPL", "COUP", "TEAM", "ZM", "DOCU", "TTD", "TTWO", "EA",
        "ASAN", "BAND", "BOX", "CLDR", "CLSK", "COUP", "CRNC", "DDOG", "DOCN", "DOMO",
        "DUOL", "EVBG", "FIVN", "FROG", "GTLB", "HUBS", "INFA", "JAMF", "MNDY", "NCNO",
        "NET", "OKTA", "PCTY", "PD", "PLAN", "QLYS", "RNG", "SAIL", "SITE", "SMAR",
        "SPT", "SUMO", "TENB", "TWLO", "U", "VEEV", "WDAY", "WIX", "ZEN", "ZS",
        
        # Santé
        "CRSP", "EDIT", "NTLA", "BEAM", "MRNA", "BNTX", "VRTX", "ALNY", "IONS", "EXEL",
        "NBIX", "ALKS", "ACAD", "ARWR", "BLUE", "CERE", "DCPH", "EXAS", "FOLD", "GERN",
        "HALO", "IDRA", "INO", "KPTI", "MGNX", "NKTR", "NVAX", "OCUL", "RARE", "SRPT",
        "ABMD", "ACAD", "ACHC", "AGIO", "ALKS", "ALNY", "AMED", "ANGI", "ATRC", "AVTR",
        "AXSM", "BHC", "BMRN", "BRKR", "CARA", "CBLI", "CDNA", "CERS", "CHMA", "CLDX",
        "CLVS", "CNMD", "COLL", "CRNX", "CRSP", "CTLT", "CUTR", "CVET", "DCPH", "DVAX",
        "EBS", "EDIT", "ENDP", "EPZM", "ESPR", "EXEL", "FOLD", "GERN", "GKOS", "GLPG",
        "GRFS", "HALO", "HOLX", "HZNP", "ICUI", "IDRA", "IMMU", "INO", "IONS", "IRWD",
        "ITCI", "JAZZ", "KPTI", "LGND", "LMAT", "MGNX", "MOR", "MRTX", "NKTR", "NVAX",
        "OCUL", "OMER", "ONCT", "PACB", "PBYI", "PDCO", "PGNX", "PRAH", "PRTA", "PTCT",
        "RARE", "RGNX", "RPRX", "SAGE", "SGEN", "SMMT", "SRPT", "STAA", "SUPN", "TNDM",
        "TROV", "UROV", "UTHR", "VCEL", "VNDA", "VRTX", "WAT", "WST", "XENE", "ZTS",
        
        # Finance
        "AFRM", "SOFI", "UPST", "LC", "LDI", "OPFI", "PROG", "RKT", "UWMC", "WOLF",
        "AX", "BHF", "CACC", "CADE", "CATY", "CFR", "CIVI", "COLB", "CVBF", "FULT",
        "ABCB", "ACGL", "AGM", "AHL", "AINV", "AMAL", "AMBC", "AMSF", "AMTB", "ANAT",
        "ARCC", "ARES", "ASB", "ASPS", "ATLC", "AUB", "AXS", "BANF", "BANR", "BBDC",
        "BCOR", "BCOV", "BHF", "BHLB", "BKU", "BL", "BLKB", "BMRC", "BNS", "BOFI",
        "BPOP", "BRKL", "BRX", "BUSE", "CACC", "CADE", "CASH", "CATY", "CBAN", "CBFV",
        "CBNK", "CBSH", "CCB", "CCNE", "CFFI", "CFR", "CHCO", "CHMG", "CIVI", "CLBK",
        "CLMS", "CNOB", "COLB", "CPF", "CPSS", "CRVL", "CSFL", "CSWC", "CTBI", "CVBF",
        "CVCY", "CVLT", "CWBC", "CWT", "CXW", "CYH", "DBD", "DCOM", "DFIN", "DHC",
        "DNBF", "EBSB", "EGBN", "EGP", "EIG", "ENVA", "ESQ", "ESSA", "EVBN", "EVTC",
        "FBC", "FBK", "FBMS", "FBNC", "FBP", "FCBP", "FCNCA", "FFBC", "FFIC", "FFIN",
        "FFWM", "FGBI", "FIBK", "FISI", "FITB", "FLIC", "FMBI", "FMBH", "FMNB", "FNB",
        "FNF", "FULT", "FUNC", "GABC", "GBCI", "GBL", "GBNK", "GCBC", "GEF", "GHL",
        "GLBZ", "GNW", "GPI", "GSBC", "HAFC", "HBNC", "HCKT", "HCSG", "HFWA", "HIFS",
        "HMN", "HOMB", "HOPE", "HTH", "HTLF", "HWBK", "IBCP", "IBOC", "IBTX", "ICBK",
        "IDCC", "INDB", "INTL", "ISBC", "ISTR", "JJSF", "KRNY", "LBAI", "LBC", "LCNB",
        "LION", "LKFN", "LPLA", "LTC", "MBI", "MBIN", "MBWM", "MCBC", "MCFT", "MCY",
        "MFIN", "MGEE", "MHLD", "MORN", "MPB", "MRTN", "MSBI", "MSEX", "MSFG", "MTB",
        "MTRN", "NATH", "NBHC", "NBN", "NCBS", "NECB", "NFBK", "NMIH", "NNI", "NRIM",
        "NWBI", "NWLI", "OCFC", "OFG", "ONB", "OPB", "ORRF", "OSBC", "OTTR", "OVBC",
        "PACW", "PB", "PBC", "PBCT", "PCH", "PEBO", "PFBC", "PFC", "PFG", "PFS",
        "PGC", "PKBK", "PLBC", "PMT", "PNFP", "PRA", "PRK", "PROV", "PUB", "PULB",
        "QCRH", "RBCAA", "RBNC", "RCII", "RDN", "RF", "RNST", "ROIC", "RUSHA", "RUSHB",
        "SASR", "SBCF", "SBNY", "SBSI", "SCHL", "SCVL", "SEIC", "SFBS", "SFNC", "SGBX",
        "SHBI", "SIGI", "SIVB", "SLM", "SMBC", "SMMF", "SNV", "SP", "SPFI", "SRCE",
        "SSB", "STBA", "STC", "STL", "STT", "SUSB", "SUSC", "SVC", "SYBT", "TBBK",
        "TCBI", "TCBK", "TFSL", "THFF", "TILE", "TIPT", "TMP", "TOWN", "TRST", "TRUP",
        "TSBK", "TSC", "TTS", "UBCP", "UBSI", "UCBI", "UCFC", "UFCS", "UMPQ", "UNTY",
        "USB", "UVSP", "VBTX", "VC", "VLY", "VNDA", "VNO", "VRSK", "WABC", "WAFD",
        "WASH", "WB", "WBS", "WETF", "WFC", "WLDN", "WLFC", "WMK", "WRLD", "WSBC",
        "WSFS", "WTBA", "WVE", "WVFC", "WW", "WWD", "WYNN", "XEL", "XL", "XLNX",
        "YORW", "ZION",

        # Industrie
        "AA", "ALB", "AMR", "ARNC", "ATI", "CENX", "CMC", "CRS", "CSTM", "KALU",
        "KALU", "NUE", "RS", "SCHN", "STLD", "WOR", "X", "ZIM", "GOGL", "MATX",
        "AAN", "ABM", "ACCO", "ADTN", "AIT", "AJRD", "AL", "ALG", "ALLE", "AME",
        "AMN", "ARCB", "ARW", "ASGN", "ASIX", "ATKR", "ATU", "AWI", "AXL", "B",
        "BCC", "BCPC", "BCO", "BECN", "BELFB", "BHE", "BMI", "BRC", "BRK.B", "BRO",
        "BWXT", "CACI", "CARR", "CAT", "CBT", "CCK", "CDW", "CE", "CFX", "CHX",
        "CIR", "CLH", "CLW", "CMC", "CNHI", "COL", "CPAC", "CR", "CRS", "CSL",
        "CTB", "CUB", "CVA", "CVCO", "CW", "CXT", "DCI", "DCO", "DD", "DOV",
        "DRQ", "DXYN", "DY", "ECL", "EME", "EMN", "ENR", "ENS", "EPAC", "ESNT",
        "ETN", "EVH", "EXPO", "FAST", "FBIN", "FELE", "FIX", "FLS", "FORM", "FOXF",
        "FTV", "GATX", "GEF", "GFF", "GGG", "GMS", "GNE", "GRA", "GTLS", "HDS",
        "HEES", "HI", "HII", "HNI", "HON", "HUBB", "HWM", "IEX", "IIIN", "IR",
        "ITW", "JBT", "JBLU", "JCI", "KALU", "KBR", "KEX", "KMT", "KNX", "LECO",
        "LFUS", "LII", "LNN", "LPG", "LZB", "MATW", "MDU", "MIDD", "MLI", "MMS",
        "MPX", "MSM", "MTX", "NBR", "NCI", "NCR", "NDSN", "NEM", "NPO", "NSC",
        "NSP", "NVEE", "NWPX", "OC", "ODC", "OGE", "OII", "OIS", "OLN", "OSK",
        "PACK", "PATK", "PBI", "PCAR", "PCH", "PCP", "PDM", "PENN", "POWL", "PPG",
        "PRI", "PRIM", "PRLB", "PRMW", "PRO", "PWR", "PXS", "R", "RBC", "RBCP",
        "RHI", "RPM", "RRX", "RSG", "RTX", "SAIA", "SCL", "SEB", "SEE", "SHW",
        "SIG", "SITE", "SJM", "SLGN", "SMG", "SPB", "SPXC", "SR", "SSD", "ST",
        "STE", "STRL", "SUM", "SWK", "SWX", "SXI", "TEX", "TKR", "TNC", "TREX",
        "TRN", "TRS", "TTC", "TTEK", "TUP", "UFPI", "UNF", "VMI", "VSH", "WCC",
        "WIRE", "WLK", "WOR", "WSO", "WTS", "WWD", "XYL", "ZEXIT",

        # Énergie
        "AR", "CHK", "CNX", "CPE", "DVN", "EOG", "FANG", "MRO", "NOV", "OXY",
        "PXD", "RRC", "SM", "SWN", "VLO", "WLL", "XOM", "CVX", "COP", "PBF",
        "AEIS", "AM", "APA", "AROC", "BKR", "BP", "BRY", "CEQP", "CHX", "CLR",
        "CNQ", "CPG", "CRK", "CTRA", "CVE", "DCP", "DKL", "DK", "DMLP", "DNR",
        "DRQ", "DVN", "ENBL", "ENLC", "ENPH", "EPD", "ET", "ETRN", "FANG", "FET",
        "FTSI", "GLNG", "GLOG", "GMLP", "GPP", "GPRK", "HAL", "HEP", "HESM", "HFC",
        "HP", "HPR", "ICD", "KOS", "LBRT", "LNG", "LPI", "LRE", "LTS", "MDR",
        "MPC", "MPLX", "MRC", "MRO", "MUR", "NBL", "NFG", "NGL", "NOV", "NRP",
        "OAS", "OGE", "OKE", "OMP", "OVV", "PAA", "PAGP", "PARR", "PBF", "PBR",
        "PDCE", "PDS", "PE", "PEG", "PENN", "PES", "PXD", "QEP", "RRC", "SBOW",
        "SBR", "SD", "SDRL", "SFL", "SGY", "SHLX", "SJT", "SLB", "SM", "SPH",
        "SPN", "SRCI", "STNG", "SUN", "SWN", "TALO", "TGS", "TRGP", "TRP", "TS",
        "TUSK", "UEC", "VET", "VLO", "WES", "WLL", "WPX", "WTI", "XEC", "XOM"
    }
    stocks.update(russell2000)
    
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
    
    # Récupérer la liste des actions du Russell 2000
    russell_stocks = set(get_stocks_list())
    
    # Fusionner les deux ensembles
    all_stocks = existing_stocks.union(russell_stocks)
    
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
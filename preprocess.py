import json
import os.path
from pathlib import Path

import pandas as pd
import pycountry

data_path: Path = Path('data')
per_year_path: Path = data_path / 'per_year'
code_mapping_path: Path = Path('codes')
preprocessed_path: Path = data_path / 'preprocessed'

INTERNATIONAL_CODES = pd.read_csv(Path("codes") / "international_codes.csv")['Code'].values
INTERNATIONAL_REGION_MAPPING = pd.read_csv(Path("codes") / "international_region_codes.csv").set_index('Code')[
    "Region"].to_dict()
INTERNATIONAL_ORGANIZATION_MAPPING = \
    pd.read_csv(Path("codes") / "international_organization_codes.csv").set_index('Code')["Organization name"].to_dict()
KEDS_MAPPING = pd.read_csv(Path("codes") / "keds_codes.csv").set_index('Code')["Organization name"].to_dict()
COUNTRIES_MAPPING = {country.alpha_3: country.name for country in pycountry.countries}

CODE_MAPPING = {}

UNK_CODES = ["UNK_CODE", "UNK_IGO"]


def load_data_year(year: int) -> pd.DataFrame:
    return pd.read_csv(preprocessed_path / f"preprocessed_{year}.csv")

def load_code_mapping() -> dict:
    global CODE_MAPPING
    try:
        with open(code_mapping_path / 'code_mapping.json', 'r') as f:
            CODE_MAPPING = json.load(f)
    except FileNotFoundError:
        print("Error in loading code mapping")
        raise FileNotFoundError


def save_code_mapping() -> None:
    # Check if exist an old mapping
    try:
        with open(code_mapping_path / 'code_mapping.json', "w") as f:
            json.dump(CODE_MAPPING, f, indent=4, sort_keys=True, ensure_ascii=False)
    except:
        print("Error in saving code mapping")



def extract_country_code(code) -> str | None:
    main_code = code[:3]
    if main_code in COUNTRIES_MAPPING:
        CODE_MAPPING[main_code] = COUNTRIES_MAPPING[main_code]

        return main_code
    else:
        return None

def extract_keds_code(code) -> str | None:
    if code in KEDS_MAPPING:
        CODE_MAPPING[code] = KEDS_MAPPING[code]
        return code
    elif (international_org_code := code[:6]) in KEDS_MAPPING:
        CODE_MAPPING[international_org_code] = KEDS_MAPPING[international_org_code]
        return international_org_code
    else:
        return None

def extract_international_org_code(code: str) -> str | None:
    if code in INTERNATIONAL_ORGANIZATION_MAPPING:
        CODE_MAPPING[code] = INTERNATIONAL_ORGANIZATION_MAPPING[code]
        return code
    else:
        return None

def extract_region_code(code: str) -> str | None:
    if code in INTERNATIONAL_REGION_MAPPING:
        CODE_MAPPING[code] = INTERNATIONAL_REGION_MAPPING[code]
        return code
    else:
        return None

def extract_international_code(x):
    if x[:3] in INTERNATIONAL_CODES:
        if (code := extract_keds_code(x)) is not None:
            return code
        elif (code := extract_international_org_code(x)) is not None:
            return code
        else:
            CODE_MAPPING["UNK_IGO"] = "UNK_IGO"
            return "UNK_IGO"
    elif (code := extract_region_code(x)) is not None:
        return code
    else:
        CODE_MAPPING["UNK_CODE"] = "UNK_CODE"
        return "UNK_CODE"


def convert_to_country_code(df: pd.DataFrame) -> pd.DataFrame:
    def convert(x):
        country_code = extract_country_code(x)
        if country_code:
            return country_code
        else:
            return extract_international_code(x)

    df['Source code'] = df['Source'].apply(convert)
    df['Target code'] = df['Target'].apply(convert)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(~df['Source code'].isin(UNK_CODES)) & (~df['Target code'].isin(UNK_CODES))]
    df = df.drop(columns=['Source', 'Target'])
    # Drop rows with nan values
    df = df.dropna()
    return df

def save_data(df: pd.DataFrame, year: int) -> None:
    df.to_csv(preprocessed_path / f"preprocessed_{year}.csv", index=False)

def preprocess_data() -> None:
    # Preprocess data in data folder
    for file in os.listdir(per_year_path):
        year = int(file.split("_")[1].split(".")[0])
        if (preprocessed_path / f"preprocessed_{year}.csv").exists():
            print(f"File preprocessed_{year}.csv already exists, skipping file {file}")
            continue
        try:
            if file.endswith(".csv"):
                df = pd.read_csv(per_year_path / file)
                df = convert_to_country_code(df)
                df_cleaned = clean_data(df)
                save_data(df_cleaned, year)
        except Exception as e:
            print(f"Error in preprocessing {file} for year {year}: {e}")
            continue


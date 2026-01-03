import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(path, sample_size=3000):
    df = pd.read_csv(path)

    # Clean columns
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    df['milage'] = df['milage'].str.replace(' mi.', '').str.replace(',', '').astype(float)

    df['clean_title'] = df['clean_title'].fillna('No')
    df.dropna(subset=['price'], inplace=True)

    # Sample for speed
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    categorical_cols = [
        'brand', 'model', 'fuel_type',
        'transmission', 'accident', 'clean_title'
    ]

    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    features = [
        'model_year', 'milage',
        'fuel_type', 'transmission',
        'accident', 'clean_title', 'brand'
    ]

    X = df[features]
    y = df['price']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42), df

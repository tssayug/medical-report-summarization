import os
import pandas as pd
import numpy as np

def calculate_quality_score(df, text_col='input', summary_col='target'):
    # 1. Word counts & Compression Ratio
    df['input_words'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    df['summary_words'] = df[summary_col].astype(str).apply(lambda x: len(x.split()))
    df['compression_ratio'] = df['summary_words'] / (df['input_words'] + 1e-5)

    # 2. Score based on ideal compression (Penalty for summaries < 5% or > 40% of report)
    # Ideal range is around 0.15 to 0.25
    df['compression_score'] = np.exp(-np.power(df['compression_ratio'] - 0.20, 2) / (2 * (0.08 ** 2)))

    # 3. Medical Section Keyword Density Score
    medical_keywords = ['history', 'impression', 'diagnosis', 'findings', 'plan', 'patient', 'hospital', 'discharge']
    def keyword_score(text):
        text_lower = str(text).lower()
        matches = sum(1 for kw in medical_keywords if kw in text_lower)
        return matches / len(medical_keywords)

    df['clinical_density'] = df[text_col].apply(keyword_score)

    # 4. Length Uniformity Score (Avoids extreme micro/macro reports)
    # Prefers reports between 80 and 500 words
    df['length_score'] = df['input_words'].apply(lambda x: 1.0 if 80 <= x <= 500 else 0.5 if 40 <= x <= 750 else 0.1)

    # 5. Composite Final Score
    df['quality_score'] = (
        0.4 * df['compression_score'] + 
        0.4 * df['clinical_density'] + 
        0.2 * df['length_score']
    )
    return df

def preprocess_data(raw_csv_path, output_dir, train_size=3000, test_size=500):
    print(f"Loading raw dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path).dropna(subset=['input', 'target'])
    
    print("Engineering features & ranking sample quality...")
    scored_df = calculate_quality_score(df)

    # Sort dataset by quality score descending
    sorted_df = scored_df.sort_values(by='quality_score', ascending=False).reset_index(drop=True)

    # Select top-tier samples
    train_df = sorted_df.iloc[:train_size]
    test_df = sorted_df.iloc[train_size : train_size + test_size]

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_3k.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_500.csv"), index=False)
    
    micro_size = min(10, len(train_df))
    train_df.head(100).to_csv(os.path.join(output_dir, "train_100_sample.csv"), index=False)

    print("\nFeature-Engineered Preprocessing complete!")
    print(f" -> Top Train subset: {len(train_df)} rows saved to '{output_dir}/train_3k.csv'")
    print(f" -> High-Quality Test subset: {len(test_df)} rows saved to '{output_dir}/test_500.csv'")
    print(f" -> Average Quality Score selected: {train_df['quality_score'].mean():.3f}")

if __name__ == "__main__":
    RAW_PATH = os.path.join("data", "raw", "raw_15k.csv")
    OUTPUT_DIR = os.path.join("data", "processed")
    preprocess_data(RAW_PATH, OUTPUT_DIR)
"""
generate_dataset.py
--------------------
Generates synthetic forensic DNA STR (Short Tandem Repeat) profiles.
In real forensics, STR profiles consist of allele values at multiple loci.
This script simulates that data for training/testing the CNN model.
"""

import numpy as np
import pandas as pd
import os

# Seed for reproducibility
np.random.seed(42)

# Common forensic STR loci used in CODIS (FBI's DNA database system)
STR_LOCI = [
    "D3S1358", "vWA", "FGA", "D8S1179", "D21S11",
    "D18S51", "D5S818", "D13S317", "D7S820", "D16S539",
    "TH01", "TPOX", "CSF1PO", "D2S1338", "D19S433"
]

# Each locus has 2 allele values (diploid organism)
# Allele values typically range from 5–40 repeats


def generate_str_profile(individual_id=None):
    """Generate a single STR profile (15 loci × 2 alleles = 30 features)."""
    profile = []
    for locus in STR_LOCI:
        allele1 = np.random.randint(5, 35)
        allele2 = np.random.randint(5, 35)
        profile.extend([allele1, allele2])
    return profile


def generate_match_pair(base_profile):
    """Generate a matching profile (same person, slight noise = lab error simulation)."""
    noise = np.random.normal(0, 0.3, len(base_profile))
    matched = np.array(base_profile, dtype=float) + noise
    matched = np.round(matched).astype(int)
    matched = np.clip(matched, 5, 35)
    return matched.tolist()


def generate_non_match_profile():
    """Generate a completely different (non-matching) profile."""
    return generate_str_profile()


def create_dataset(n_pairs=2000):
    """
    Create a balanced dataset of:
    - Label 1 (Match): Two profiles from the same individual
    - Label 0 (No Match): Two profiles from different individuals
    """
    records = []

    for i in range(n_pairs):
        # MATCH pair
        base = generate_str_profile(individual_id=i)
        matched = generate_match_pair(base)
        combined_match = base + matched
        records.append(combined_match + [1])  # label = 1 (match)

        # NO MATCH pair
        person_a = generate_str_profile()
        person_b = generate_non_match_profile()
        combined_no_match = person_a + person_b
        records.append(combined_no_match + [0])  # label = 0 (no match)

    # Build column names
    profile1_cols = [f"P1_{locus}_{allele}" for locus in STR_LOCI for allele in ["A1", "A2"]]
    profile2_cols = [f"P2_{locus}_{allele}" for locus in STR_LOCI for allele in ["A1", "A2"]]
    columns = profile1_cols + profile2_cols + ["label"]

    df = pd.DataFrame(records, columns=columns)
    return df


if __name__ == "__main__":
    print("Generating synthetic forensic DNA STR dataset...")
    df = create_dataset(n_pairs=2000)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dna_str_dataset.csv", index=False)

    print(f"Dataset saved: data/dna_str_dataset.csv")
    print(f"Total samples : {len(df)}")
    print(f"Match samples : {df['label'].sum()}")
    print(f"No-Match      : {len(df) - df['label'].sum()}")
    print(f"Features      : {len(df.columns) - 1}")
    print("\nSample preview:")
    print(df.head(3))

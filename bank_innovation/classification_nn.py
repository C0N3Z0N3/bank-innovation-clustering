import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer  # NOT StandardScaler — data is heavy-tailed
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# SETTINGS
# ============================================================
CSV_PATH = "data/bank_innovation_clustered_optimized.csv"
BANK_ID_COL = "rssd9017"
TIER_COL = "bank_tier"
LABEL_COL = "innovation_cluster"
TIME_COLS = ["year_from", "year_to"]

FEATURE_COLS = [
    "tech_investment_ratio_change",
    "nib_deposit_ratio_change",
    "service_charge_intensity_change",
    "efficiency_ratio_change",
    "nonint_income_pct_change",
    "loans_to_assets_change",
    "equity_to_assets_change",
    "deposits_to_assets_change",
    "roa_change",
    "roe_change",
    "nontrans_deposits_pct_change",
    "digital_revenue_ratio_change",
    "non_branch_revenue_pct_change",
    "loan_yield_change",
    "securities_to_assets_change",
    "expense_per_salary_dollar_change",
    "occupancy_intensity_change",
    "chargeoff_rate_change",
    "provision_intensity_change",
    "asset_growth_capacity_change",
]

# Valid tier names in the data
VALID_TIERS = ["Small", "Medium", "Large"]


# ============================================================
# 1. LOAD AND CLEAN DATA
# ============================================================
def load_clustered_data(
    csv_path=CSV_PATH,
    bank_id_col=BANK_ID_COL,
    tier_col=TIER_COL,
    label_col=LABEL_COL,
    time_cols=TIME_COLS,
    feature_cols=FEATURE_COLS,
    tier=None,          # e.g. "Medium" — ALWAYS pass this, never mix tiers
    drop_noise=True,
):
    """
    Load clustered bank data, filter to one tier, and do basic cleaning.

    Always filter to a single tier before training. Mixing tiers means
    cluster labels are not comparable (cluster 2 means something different
    in Small vs Large banks).
    """
    df = pd.read_csv(csv_path)

    needed_cols = [bank_id_col, tier_col, label_col] + time_cols + feature_cols
    df = df[needed_cols].copy()

    # Filter to one tier — this is required
    if tier is not None:
        if tier not in VALID_TIERS:
            raise ValueError(f"tier must be one of {VALID_TIERS}, got '{tier}'")
        df = df[df[tier_col] == tier].copy()
        print(f"Filtered to tier: {tier} ({len(df)} rows)")
    else:
        print("WARNING: No tier specified — all tiers mixed together. This is probably wrong.")

    df = df.dropna(subset=[bank_id_col, label_col])

    # Drop noise points (cluster label -1) — they don't have a meaningful target
    if drop_noise:
        before = len(df)
        df = df[df[label_col] != -1].copy()
        print(f"Dropped {before - len(df)} noise rows")

    # Fill any remaining NaN features with 0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Sort so sequences are in chronological order per bank
    df = df.sort_values([bank_id_col] + time_cols).reset_index(drop=True)

    print(f"Final data shape: {df.shape}")
    print("Cluster counts:")
    print(df[label_col].value_counts().sort_index())

    return df


# ============================================================
# 2. ENCODE LABELS
# ============================================================
def encode_labels(df, label_col=LABEL_COL):
    """
    Map cluster labels to 0, 1, 2, ... for model training.
    Returns the dataframe with a new 'label_encoded' column,
    plus mappings to convert back and forth.
    """
    df = df.copy()

    unique_labels = sorted(df[label_col].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_label_map = {idx: label for label, idx in label_map.items()}

    df["label_encoded"] = df[label_col].map(label_map)

    print("Label map:", label_map)
    print("Number of classes:", len(label_map))

    return df, label_map, reverse_label_map


# ============================================================
# 3. SPLIT BY BANK
# ============================================================
def split_by_bank(
    df,
    bank_id_col=BANK_ID_COL,
    test_size=0.2,
    random_state=42,
):
    """
    Split at the bank level, not the row level.

    This prevents leakage — without this, the same bank's windows
    could appear in both train and test, which inflates accuracy.
    """
    unique_banks = df[bank_id_col].unique()

    train_banks, test_banks = train_test_split(
        unique_banks,
        test_size=test_size,
        random_state=random_state,
    )

    train_df = df[df[bank_id_col].isin(train_banks)].copy()
    test_df = df[df[bank_id_col].isin(test_banks)].copy()

    print(f"Train banks: {len(train_banks)} | Test banks: {len(test_banks)}")

    return train_df, test_df


# ============================================================
# 4. SCALE FEATURES
# ============================================================
def scale_features(
    train_df,
    test_df,
    feature_cols=FEATURE_COLS,
    n_quantiles=1000,
):
    """
    Scale features using QuantileTransformer fit on training data only.

    We use QuantileTransformer (NOT StandardScaler) because the change
    score distributions are extremely heavy-tailed (skewness up to 223,
    kurtosis up to 49,708). StandardScaler would compress 99%+ of
    observations into a tiny range near zero, killing model performance.

    QuantileTransformer maps each feature to a uniform distribution,
    giving the LSTM actual variation to learn from.
    """
    scaler = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="uniform",
        random_state=42,
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, scaler


# ============================================================
# 5. BUILD SEQUENCES PER BANK
# ============================================================
def build_sequences(
    dataframe,
    bank_id_col=BANK_ID_COL,
    time_cols=TIME_COLS,
    feature_cols=FEATURE_COLS,
    label_encoded_col="label_encoded",
):
    """
    Build one time series sequence per bank.

    Each step in the sequence = one 3-year rolling window's change scores.
    A bank with 9 windows gets a sequence of length 9.
    A bank with 3 windows gets a sequence of length 3.

    Label = the cluster of the LAST window (most recent behavior).
    This is more meaningful than the first window — we're asking
    "given a bank's full history, what cluster did it end up in?"
    """
    sequences = []
    labels = []
    bank_ids = []

    for bank_id, group in dataframe.groupby(bank_id_col):
        group = group.sort_values(time_cols)

        # Shape: (num_windows, num_features)
        seq = group[feature_cols].values.astype("float32")

        # Use the LAST window's cluster label as the target
        label = int(group[label_encoded_col].iloc[-1])

        sequences.append(seq)
        labels.append(label)
        bank_ids.append(bank_id)

    labels = np.array(labels, dtype="int32")

    print(f"Built {len(sequences)} sequences")
    print(f"Sequence length range: {min(len(s) for s in sequences)} – {max(len(s) for s in sequences)}")

    return sequences, labels, bank_ids


# ============================================================
# 6. PAD SEQUENCES
# ============================================================
def pad_sequence_sets(train_sequences, test_sequences):
    """
    Pad sequences to the same length so they can be batched.

    Shorter sequences get zeros appended at the end (post-padding).
    The Masking layer in the model tells the LSTM to ignore those zeros.
    """
    all_lengths = [len(s) for s in train_sequences] + [len(s) for s in test_sequences]
    max_len = max(all_lengths)

    X_train = pad_sequences(
        train_sequences,
        maxlen=max_len,
        padding="post",
        dtype="float32",
    )

    X_test = pad_sequences(
        test_sequences,
        maxlen=max_len,
        padding="post",
        dtype="float32",
    )

    print(f"Padded to max length: {max_len}")

    return X_train, X_test, max_len


# ============================================================
# 7. COMPUTE CLASS WEIGHTS
# ============================================================
def get_class_weights(y_train):
    """
    Compute class weights to handle Steady Operators dominance.

    Without this, the model learns to predict Steady Operators every
    time and gets ~85% accuracy while being completely useless on
    the interesting minority clusters.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight_dict = dict(zip(classes, weights))
    print("Class weights:", class_weight_dict)
    return class_weight_dict


# ============================================================
# 8. BUILD LSTM MODEL
# ============================================================
def build_lstm_model(input_timesteps, input_features, num_classes):
    """
    LSTM classifier.

    Masking layer ignores the padded zeros at the end of short sequences.
    Dropout reduces overfitting on the small minority clusters.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_timesteps, input_features)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# 9. BUILD GRU MODEL (comparison)
# ============================================================
def build_gru_model(input_timesteps, input_features, num_classes):
    """
    GRU classifier — same structure as LSTM for fair comparison.

    GRUs are faster and sometimes outperform LSTMs on shorter sequences.
    With max ~9 timesteps here, GRU may actually do better.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_timesteps, input_features)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.GRU(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# 10. FULL PREP PIPELINE
# ============================================================
def prepare_lstm_data(
    csv_path=CSV_PATH,
    bank_id_col=BANK_ID_COL,
    tier_col=TIER_COL,
    label_col=LABEL_COL,
    time_cols=TIME_COLS,
    feature_cols=FEATURE_COLS,
    tier="Medium",      # default to Medium — change to "Small" or "Large" as needed
    drop_noise=True,
    test_size=0.2,
    random_state=42,
):
    """
    Full preprocessing pipeline. Returns a dict with everything
    needed to train and evaluate the model.

    Parameters
    ----------
    tier : str
        Which bank tier to model. One of "Small", "Medium", "Large".
        Always run one tier at a time — clusters are not comparable
        across tiers.
    """
    df = load_clustered_data(
        csv_path=csv_path,
        bank_id_col=bank_id_col,
        tier_col=tier_col,
        label_col=label_col,
        time_cols=time_cols,
        feature_cols=feature_cols,
        tier=tier,
        drop_noise=drop_noise,
    )

    df, label_map, reverse_label_map = encode_labels(df, label_col=label_col)

    train_df, test_df = split_by_bank(
        df,
        bank_id_col=bank_id_col,
        test_size=test_size,
        random_state=random_state,
    )

    train_df, test_df, scaler = scale_features(
        train_df,
        test_df,
        feature_cols=feature_cols,
    )

    train_sequences, y_train, train_bank_ids = build_sequences(
        train_df,
        bank_id_col=bank_id_col,
        time_cols=time_cols,
        feature_cols=feature_cols,
    )

    test_sequences, y_test, test_bank_ids = build_sequences(
        test_df,
        bank_id_col=bank_id_col,
        time_cols=time_cols,
        feature_cols=feature_cols,
    )

    X_train, X_test, max_len = pad_sequence_sets(train_sequences, test_sequences)

    class_weights = get_class_weights(y_train)

    print("\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_map": label_map,
        "reverse_label_map": reverse_label_map,
        "scaler": scaler,
        "max_len": max_len,
        "class_weights": class_weights,
        "train_bank_ids": train_bank_ids,
        "test_bank_ids": test_bank_ids,
        "tier": tier,
    }
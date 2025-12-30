"""
Skin Cancer Model Training Script

This script trains the MobileNet + LSTM model for skin lesion classification
and saves the weights for use in the Smart Mirror system.

Prerequisites:
1. Download ISIC 2020 dataset images to: data/train/
2. Ensure metadata_filtered.csv is in the data/ folder

Usage:
    python train_model.py --data-dir /path/to/data --output-dir ./models

The trained model will be saved as:
    - skin_cancer_lstm.h5 (Keras format)
    - skin_cancer_lstm.keras (Keras 3 format)
    - feature_extractor metadata
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_data(data_dir: str, metadata_file: str = "metadata_filtered.csv"):
    """Load and prepare the dataset."""
    metadata_path = os.path.join(data_dir, metadata_file)
    df = pd.read_csv(metadata_path)

    # Map labels
    df['target'] = df['benign_malignant'].map({'benign': 0, 'malignant': 1})

    print(f"Loaded {len(df)} records")
    print(f"Unique patients: {df['patient_id'].nunique()}")

    return df


def create_sequences(df, data_dir: str, feature_extractor):
    """Create temporal sequences for LSTM training."""
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications.mobilenet import preprocess_input

    train_dir = os.path.join(data_dir, "train")
    unique_patients = df['patient_id'].unique()

    sequences = []
    labels = []

    print("Creating sequences from patient data...")

    for patient_id in unique_patients:
        patient_df = df[df['patient_id'] == patient_id]

        for anatom_site in patient_df['anatom_site_general_challenge'].unique():
            site_df = patient_df[patient_df['anatom_site_general_challenge'] == anatom_site].copy()
            site_df.sort_values(by=['age_approx', 'target'], ascending=[True, False], inplace=True)

            first_by_age = site_df.groupby('age_approx').first().reset_index()

            if len(first_by_age) > 1:
                # Keep first 2 images
                rows = first_by_age.head(2)

                seq_features = []
                seq_labels = []

                for _, row in rows.iterrows():
                    img_path = os.path.join(train_dir, f"{row['image_name']}.jpg")

                    if os.path.exists(img_path):
                        img = load_img(img_path, target_size=(224, 224))
                        img_array = img_to_array(img)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_preprocessed = preprocess_input(img_array)

                        features = feature_extractor.predict(img_preprocessed, verbose=0)
                        seq_features.append(np.squeeze(features))
                        seq_labels.append(row['target'])

                if len(seq_features) == 2:
                    sequences.append(seq_features)
                    labels.append(seq_labels[-1])  # Final label

    print(f"Created {len(sequences)} sequences")
    return sequences, labels


def balance_dataset(sequences, labels, melanoma_ratio=0.6):
    """Balance dataset to have specified ratio of melanoma cases."""
    sequences = np.array(sequences, dtype=object)
    labels = np.array(labels)

    melanoma_idx = np.where(labels == 1)[0]
    benign_idx = np.where(labels == 0)[0]

    # Calculate how many benign samples to keep
    n_benign = int(len(melanoma_idx) * (1 - melanoma_ratio) / melanoma_ratio)
    n_benign = min(n_benign, len(benign_idx))

    np.random.seed(42)
    selected_benign = np.random.choice(benign_idx, n_benign, replace=False)

    selected_idx = np.concatenate([melanoma_idx, selected_benign])
    np.random.shuffle(selected_idx)

    balanced_sequences = [sequences[i] for i in selected_idx]
    balanced_labels = labels[selected_idx]

    print(f"Balanced dataset: {len(balanced_sequences)} samples")
    print(f"  Melanoma: {sum(balanced_labels == 1)}")
    print(f"  Benign: {sum(balanced_labels == 0)}")

    return balanced_sequences, balanced_labels


def create_feature_extractor():
    """Create MobileNet feature extractor."""
    from tensorflow.keras.applications import MobileNet
    from tensorflow.keras.models import Model

    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(224, 224, 3)
    )

    return Model(inputs=base_model.input, outputs=base_model.output)


def create_lstm_model(input_shape):
    """Create LSTM classifier."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Masking
    from tensorflow.keras.metrics import Precision, Recall

    model = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        LSTM(64, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )

    return model


def train_model(X_train, y_train, X_val, y_val, epochs=60):
    """Train the LSTM model."""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def save_model(model, output_dir: str):
    """Save the trained model in multiple formats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save in Keras format
    keras_path = output_path / "skin_cancer_lstm.keras"
    model.save(keras_path)
    print(f"Saved Keras model to: {keras_path}")

    # Save weights only
    weights_path = output_path / "skin_cancer_lstm_weights.h5"
    model.save_weights(weights_path)
    print(f"Saved weights to: {weights_path}")

    # Save model config
    config = {
        "model_type": "MobileNet_LSTM",
        "feature_extractor": "MobileNet",
        "feature_dim": 1024,
        "sequence_length": 2,
        "lstm_units": 64,
        "input_shape": [2, 1024]
    }

    config_path = output_path / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Train skin cancer detection model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory with train/ and metadata_filtered.csv")
    parser.add_argument("--output-dir", type=str, default="./", help="Output directory for saved model")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test set ratio")
    args = parser.parse_args()

    print("=" * 60)
    print("Skin Cancer Model Training")
    print("=" * 60)

    # Import TensorFlow
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")

    # Load data
    df = load_data(args.data_dir)

    # Create feature extractor
    print("\nLoading MobileNet feature extractor...")
    feature_extractor = create_feature_extractor()
    print(f"Feature extractor output shape: {feature_extractor.output_shape}")

    # Create sequences
    sequences, labels = create_sequences(df, args.data_dir, feature_extractor)

    if len(sequences) == 0:
        print("ERROR: No sequences created. Check that images exist in data/train/")
        return

    # Balance dataset
    sequences, labels = balance_dataset(sequences, labels)

    # Pad sequences
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    X = pad_sequences(sequences, padding='post', dtype='float32', maxlen=2)
    y = np.array(labels)

    print(f"\nData shape: X={X.shape}, y={y.shape}")

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=args.test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train
    print("\nTraining model...")
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=args.epochs)

    # Evaluate
    print("\nEvaluating on test set...")
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")

    # Save
    print("\nSaving model...")
    save_model(model, args.output_dir)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

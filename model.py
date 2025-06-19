import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load and check the dataset"""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({(df['Class'].sum()/len(df))*100:.2f}%)")
    return df

def preprocess_data(df):
    """Preprocess the data"""
    # Split features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle imbalanced data with SMOTE
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"Training samples: {len(X_train_balanced)}")
    print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, class_weight_dict

def build_model(input_dim):
    """Build neural network model"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, class_weights):
    """Train the model"""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val_split, y_val_split),
        epochs=50,
        batch_size=512,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n=== MODEL EVALUATION ===")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return roc_auc, y_pred, y_pred_proba

def main():
    """Main execution function"""
    print("=== CREDIT CARD FRAUD DETECTION ===\n")
    
    # Load data
    df = load_data('D:\CREDIT CARD FRAUD DETECTION\CREDIT-CARD-FRAUD-DETECTION\ccenv\Dataset\creditcard.csv')  # Replace with your file path
    
    # Preprocess
    X_train, X_test, y_train, y_test, class_weights = preprocess_data(df)
    
    # Build model
    model = build_model(X_train.shape[1])
    print(f"\nModel built with {X_train.shape[1]} input features")
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, X_train, y_train, class_weights)
    
    # Evaluate model
    roc_auc, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Save model
    model.save('fraud_detection_model.h5')
    print(f"\nModel saved as 'fraud_detection_model.h5'")
    print(f"Final ROC-AUC Score: {roc_auc:.4f}")
    
    return model, roc_auc

if __name__ == "__main__":
    model, score = main()
#!/usr/bin/env python3
"""
Deep Learning Models for Phishing Detection
Phase 3: Model Development and Initial Testing
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertModel, BertTokenizer, TFDistilBertModel, DistilBertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple, Optional
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DeepLearningModels:
    """Deep learning models for phishing detection."""
    
    def __init__(self, max_sequence_length: int = 500, 
                 embedding_dim: int = 100,
                 vocab_size: int = 10000,
                 random_state: int = 42):
        """Initialize deep learning models."""
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.models = {}
        self.tokenizer = None
        self.bert_tokenizer = None
        self.model_scores = {}
        
    def build_lstm_model(self, input_shape: Tuple) -> keras.Model:
        """Build LSTM model for text classification."""
        model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, 
                           input_length=self.max_sequence_length),
            layers.LSTM(128, dropout=0.5, recurrent_dropout=0.5, 
                       return_sequences=True),
            layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall()]
        )
        
        return model
    
    def build_bidirectional_lstm(self, input_shape: Tuple) -> keras.Model:
        """Build Bidirectional LSTM model."""
        model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim,
                           input_length=self.max_sequence_length),
            layers.Bidirectional(layers.LSTM(128, dropout=0.5, 
                                           recurrent_dropout=0.5,
                                           return_sequences=True)),
            layers.Bidirectional(layers.LSTM(64, dropout=0.5,
                                           recurrent_dropout=0.5)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()]
        )
        
        return model
    
    def build_cnn_lstm_model(self, input_shape: Tuple) -> keras.Model:
        """Build CNN-LSTM hybrid model."""
        model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim,
                           input_length=self.max_sequence_length),
            layers.Conv1D(128, 5, activation='relu'),
            layers.MaxPooling1D(5),
            layers.Conv1D(128, 5, activation='relu'),
            layers.MaxPooling1D(5),
            layers.LSTM(128, dropout=0.5, recurrent_dropout=0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()]
        )
        
        return model
    
    def build_gru_model(self, input_shape: Tuple) -> keras.Model:
        """Build GRU model for text classification."""
        model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim,
                           input_length=self.max_sequence_length),
            layers.GRU(128, dropout=0.5, recurrent_dropout=0.5,
                      return_sequences=True),
            layers.GRU(64, dropout=0.5, recurrent_dropout=0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()]
        )
        
        return model
    
    def build_transformer_model(self, input_shape: Tuple) -> keras.Model:
        """Build a simple Transformer model."""
        inputs = layers.Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        embedding = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        
        # Multi-head attention
        attention = layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=self.embedding_dim
        )(embedding, embedding)
        
        # Add & Norm
        attention = layers.LayerNormalization(epsilon=1e-6)(attention + embedding)
        
        # Feed Forward
        ff = layers.Dense(256, activation='relu')(attention)
        ff = layers.Dense(self.embedding_dim)(ff)
        
        # Add & Norm
        ff = layers.LayerNormalization(epsilon=1e-6)(ff + attention)
        
        # Global pooling
        pooling = layers.GlobalAveragePooling1D()(ff)
        
        # Classification layers
        dense = layers.Dense(64, activation='relu')(pooling)
        dense = layers.Dropout(0.5)(dense)
        outputs = layers.Dense(1, activation='sigmoid')(dense)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()]
        )
        
        return model
    
    def prepare_text_data(self, texts: list, labels: np.ndarray = None,
                         fit_tokenizer: bool = True) -> Tuple:
        """Prepare text data for deep learning models."""
        if fit_tokenizer or self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size, 
                                      oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, 
                                        maxlen=self.max_sequence_length,
                                        padding='post',
                                        truncating='post')
        
        if labels is not None:
            return padded_sequences, labels
        return padded_sequences
    
    def train_model(self, model_name: str, model: keras.Model,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   epochs: int = 10, batch_size: int = 32) -> Dict:
        """Train a deep learning model."""
        print(f"\nTraining {model_name}...")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=3,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Store model
        self.models[model_name] = model
        
        # Evaluate
        train_pred = (model.predict(X_train) > 0.5).astype(int).flatten()
        train_scores = self._calculate_metrics(y_train, train_pred)
        
        val_scores = {}
        if X_val is not None and y_val is not None:
            val_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
            val_scores = self._calculate_metrics(y_val, val_pred)
        
        # Store scores
        self.model_scores[model_name] = {
            'train': train_scores,
            'validation': val_scores,
            'history': history.history
        }
        
        print(f"  Training Accuracy: {train_scores['accuracy']:.4f}")
        if val_scores:
            print(f"  Validation Accuracy: {val_scores['accuracy']:.4f}")
        
        return {
            'model': model,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'history': history.history
        }
    
    def build_bert_classifier(self, num_classes: int = 1) -> keras.Model:
        """Build BERT-based classifier (using DistilBERT for efficiency)."""
        try:
            # Use DistilBERT for efficiency
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
            
            # Input layers
            input_ids = layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
            attention_mask = layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
            
            # BERT outputs
            bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
            
            # Use pooled output
            pooled_output = bert_outputs.last_hidden_state[:, 0, :]
            
            # Classification layers
            dense = layers.Dense(64, activation='relu')(pooled_output)
            dense = layers.Dropout(0.5)(dense)
            outputs = layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(dense)
            
            model = keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
            
            model.compile(
                optimizer=optimizers.Adam(learning_rate=2e-5),
                loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            print(f"Error building BERT model: {str(e)}")
            print("Make sure transformers library is installed: pip install transformers")
            return None
    
    def prepare_bert_data(self, texts: list, max_length: int = 128) -> Dict:
        """Prepare text data for BERT model."""
        if self.bert_tokenizer is None:
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        encoded = self.bert_tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='tf'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary')
        }
    
    def save_models(self, output_dir: str = 'models/deep_learning'):
        """Save all trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = output_path / f"{model_name}"
            model.save(model_path)
            print(f"  Saved {model_name} to {model_path}")
        
        # Save tokenizer
        if self.tokenizer:
            tokenizer_path = output_path / "tokenizer.json"
            tokenizer_json = self.tokenizer.to_json()
            with open(tokenizer_path, 'w') as f:
                f.write(tokenizer_json)
            print(f"  Saved tokenizer to {tokenizer_path}")
        
        # Save scores
        scores_path = output_path / "model_scores.json"
        # Convert numpy arrays to lists for JSON serialization
        serializable_scores = {}
        for model_name, scores in self.model_scores.items():
            serializable_scores[model_name] = {
                'train': scores['train'],
                'validation': scores['validation']
            }
        
        with open(scores_path, 'w') as f:
            json.dump(serializable_scores, f, indent=2)
        print(f"  Saved scores to {scores_path}")


def main():
    """Test deep learning models."""
    print("\n" + "="*70)
    print("DEEP LEARNING MODELS TEST")
    print("="*70)
    
    # Generate dummy text data
    sample_texts = [
        "Your account has been suspended. Click here to verify.",
        "Meeting scheduled for tomorrow at 2 PM.",
        "Urgent: Update your payment information immediately.",
        "Thanks for your email. I'll get back to you soon.",
        "Security alert: Suspicious activity detected on your account."
    ] * 200  # Repeat for more samples
    
    labels = np.array([1, 0, 1, 0, 1] * 200)  # 1 for phishing, 0 for legitimate
    
    # Initialize deep learning models
    dl_models = DeepLearningModels()
    
    # Prepare data
    X, y = dl_models.prepare_text_data(sample_texts, labels)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Build and train LSTM model
    lstm_model = dl_models.build_lstm_model((dl_models.max_sequence_length,))
    result = dl_models.train_model('lstm', lstm_model, X_train, y_train, 
                                  X_val, y_val, epochs=2)
    
    print("\n✓ Deep learning models test complete")


if __name__ == "__main__":
    main()
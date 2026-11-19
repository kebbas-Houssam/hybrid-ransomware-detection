"""PSO-optimized hybrid model: CNN-LSTM-Attention (stage 1) + XGBoost (stage 2)."""

import os
import traceback

import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC

from training import data as data_utils
from training.layers import AttentionLayer, F2Score
from training.pso_optimizer import PSOXGBoostOptimizer

# XGBoost imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")


class PSOHybridRansomwareDetectionModel:
    """
    Enhanced Hybrid Model with PSO-optimized XGBoost
    """

    def __init__(self, time_steps, n_features=13, lstm_units=128,
                 cnn_filters=64, attention_dim=64, dropout_rate=0.3,
                 output_dir='.'):
        self.time_steps = time_steps
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.attention_dim = attention_dim
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = None
        self.class_weights = None

        # PSO and XGBoost components
        self.pso_optimizer = None
        self.xgb_model = None
        self.xgb_scaler = None
        self.best_xgb_params = None
        self.use_xgboost = XGBOOST_AVAILABLE
        self.pso_history = None

        # Where checkpoints / plots are written during training
        self.output_dir = output_dir

    def build_model(self):
        """Build the CNN-LSTM-Attention architecture (same as before)"""
        inputs = layers.Input(shape=(self.time_steps, self.n_features),
                              name='io_trace_input')

        # CNN Feature Extraction Block
        cnn_1 = layers.Conv1D(filters=self.cnn_filters, kernel_size=3,
                              activation='relu', padding='same', name='cnn_1')(inputs)
        cnn_1 = layers.BatchNormalization(name='bn_cnn_1')(cnn_1)
        cnn_1 = layers.Dropout(self.dropout_rate, name='dropout_cnn_1')(cnn_1)

        cnn_2 = layers.Conv1D(filters=self.cnn_filters, kernel_size=5,
                              activation='relu', padding='same', name='cnn_2')(cnn_1)
        cnn_2 = layers.BatchNormalization(name='bn_cnn_2')(cnn_2)
        cnn_2 = layers.Dropout(self.dropout_rate, name='dropout_cnn_2')(cnn_2)

        cnn_3 = layers.Conv1D(filters=self.cnn_filters, kernel_size=7,
                              activation='relu', padding='same', name='cnn_3')(cnn_2)
        cnn_3 = layers.BatchNormalization(name='bn_cnn_3')(cnn_3)

        # LSTM blocks
        lstm_1 = layers.LSTM(self.lstm_units, return_sequences=True,
                             name='lstm_block_1')(cnn_3)
        lstm_1 = layers.Dropout(self.dropout_rate, name='dropout_lstm_1')(lstm_1)

        lstm_2 = layers.LSTM(self.lstm_units, return_sequences=True,
                             name='lstm_block_2')(cnn_3)
        lstm_2 = layers.Dropout(self.dropout_rate, name='dropout_lstm_2')(lstm_2)

        lstm_3 = layers.LSTM(self.lstm_units, return_sequences=True,
                             name='lstm_block_3')(cnn_3)
        lstm_3 = layers.Dropout(self.dropout_rate, name='dropout_lstm_3')(lstm_3)

        # Attention Mechanism
        attention_1, attention_weights_1 = AttentionLayer(
            attention_dim=self.attention_dim, name='attention_1')(lstm_1)
        attention_2, attention_weights_2 = AttentionLayer(
            attention_dim=self.attention_dim, name='attention_2')(lstm_2)
        attention_3, attention_weights_3 = AttentionLayer(
            attention_dim=self.attention_dim, name='attention_3')(lstm_3)

        # Concatenate attention outputs
        concatenated = layers.Concatenate(name='concat_attention')(
            [attention_1, attention_2, attention_3])

        # Classification Head
        dense_1 = layers.Dense(256, activation='relu', name='dense_1')(concatenated)
        dense_1 = layers.BatchNormalization(name='bn_dense_1')(dense_1)
        dense_1 = layers.Dropout(self.dropout_rate, name='dropout_dense_1')(dense_1)

        dense_2 = layers.Dense(128, activation='relu', name='dense_2')(dense_1)
        dense_2 = layers.BatchNormalization(name='bn_dense_2')(dense_2)
        dense_2 = layers.Dropout(self.dropout_rate / 2, name='dropout_dense_2')(dense_2)

        dense_3 = layers.Dense(64, activation='relu', name='dense_3')(dense_2)
        dense_3 = layers.Dropout(self.dropout_rate / 2, name='dropout_dense_3')(dense_3)

        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='ransomware_prediction')(dense_3)

        # Create model
        self.model = Model(inputs=inputs, outputs=output, name='iCNN_LSTM_Attention_Ransomware')

        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compile model with TensorFlow 2.10 compatible settings"""
        optimizer = keras.optimizers.legacy.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )

        metrics_list = [
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            F2Score(name='f2_score')
        ]

        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=metrics_list
        )

        return self.model

    # Data helpers live in training/data.py; kept as methods for API compatibility.
    def prepare_data(self, df, sequence_length=50, overlap=0.5):
        return data_utils.prepare_data(df, sequence_length=sequence_length, overlap=overlap)

    def handle_class_imbalance(self, y_train):
        return data_utils.handle_class_imbalance(y_train)

    def extract_hybrid_features(self, X_sequences, y_pred_proba_primary, verbose=True):
        return data_utils.extract_hybrid_features(X_sequences, y_pred_proba_primary, verbose=verbose)

    def train_pso_optimized_xgboost(self, X_hybrid_train, y_train, X_hybrid_val, y_val,
                                    n_particles=20, max_iterations=30):
        """
        Train XGBoost with PSO-optimized hyperparameters
        """
        if not self.use_xgboost:
            print("XGBoost not available, skipping PSO optimization")
            return None

        try:
            print("\n" + "="*60)
            print("PSO-OPTIMIZED XGBOOST TRAINING")
            print("="*60)

            # Scale hybrid features
            self.xgb_scaler = RobustScaler()
            X_hybrid_train_scaled = self.xgb_scaler.fit_transform(X_hybrid_train)
            X_hybrid_val_scaled = self.xgb_scaler.transform(X_hybrid_val)

            # Initialize PSO optimizer
            self.pso_optimizer = PSOXGBoostOptimizer(
                n_particles=n_particles,
                max_iterations=max_iterations
            )

            # Run PSO optimization
            best_params, fitness_history = self.pso_optimizer.optimize(
                X_hybrid_train_scaled, y_train,
                X_hybrid_val_scaled, y_val
            )

            self.best_xgb_params = best_params
            self.pso_history = fitness_history

            # Train final XGBoost model with best parameters
            print(f"\nTraining final XGBoost model with PSO-optimized parameters...")
            self.xgb_model = xgb.XGBClassifier(**best_params)

            # Train final model
            self.xgb_model.fit(
                X_hybrid_train_scaled, y_train,
                eval_set=[(X_hybrid_val_scaled, y_val)],
                verbose=True
            )

            # Plot convergence
            self.pso_optimizer.plot_convergence(
                os.path.join(self.output_dir, 'pso_convergence_xgboost.png'))

            print("PSO-optimized XGBoost training completed!")

            return self.xgb_model

        except Exception as e:
            print(f"Error in PSO XGBoost training: {e}")
            traceback.print_exc()
            return None

    def train_model(self, X_train, y_train, X_val, y_val,
                    epochs=100, batch_size=32, verbose=1,
                    pso_particles=20, pso_iterations=30):
        """
        Train the PSO-optimized hybrid model
        """
        try:
            # Validate input data
            print("Validating training data...")
            if not np.isfinite(X_train).all():
                print("Warning: Non-finite values in training data")
                X_train = np.nan_to_num(X_train, nan=0.0)

            if not np.isfinite(X_val).all():
                print("Warning: Non-finite values in validation data")
                X_val = np.nan_to_num(X_val, nan=0.0)

            class_weight_dict = self.handle_class_imbalance(y_train)

            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_f2_score',
                    patience=15,
                    restore_best_weights=True,
                    mode='max',
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_f2_score',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    mode='max',
                    verbose=1
                ),
                ModelCheckpoint(
                    os.path.join(self.output_dir, 'best_pso_ransomware_model.h5'),
                    monitor='val_f2_score',
                    mode='max',
                    save_best_only=True,
                    verbose=1
                )
            ]

            print("Starting primary CNN-LSTM-Attention training...")

            # Train primary model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=min(batch_size, 16),
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=verbose,
                validation_freq=1,
                use_multiprocessing=False,
                workers=1
            )

            # PSO-OPTIMIZED HYBRID STAGE
            if self.use_xgboost:
                print("\n" + "="*60)
                print("STARTING PSO-OPTIMIZED HYBRID STAGE")
                print("="*60)

                # Get primary model predictions
                print("Getting primary model predictions...")
                y_pred_proba_train = self.model.predict(X_train, verbose=0, batch_size=16)
                y_pred_proba_val = self.model.predict(X_val, verbose=0, batch_size=16)

                # Extract hybrid features
                X_hybrid_train = self.extract_hybrid_features(X_train, y_pred_proba_train)
                X_hybrid_val = self.extract_hybrid_features(X_val, y_pred_proba_val)

                # Train PSO-optimized XGBoost classifier
                self.train_pso_optimized_xgboost(
                    X_hybrid_train, y_train,
                    X_hybrid_val, y_val,
                    n_particles=pso_particles,
                    max_iterations=pso_iterations
                )

                print("PSO-optimized hybrid training completed!")
            else:
                print("XGBoost not available - using primary model only")

            return history

        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            raise

    def predict_pso_hybrid(self, X_test, verbose=True):
        """
        Make predictions using the PSO-optimized hybrid approach.

        ``X_test`` must already be scaled with ``self.scaler``.
        Returns ``(y_pred_proba_primary, y_pred_proba_hybrid)``, both shaped (n, 1).
        """
        try:
            # Primary model predictions
            y_pred_proba_primary = self.model.predict(X_test, verbose=0, batch_size=16)

            if not self.use_xgboost or self.xgb_model is None:
                print("Using primary model predictions only")
                return y_pred_proba_primary, y_pred_proba_primary

            # Extract hybrid features for PSO-optimized XGBoost
            X_hybrid_test = self.extract_hybrid_features(X_test, y_pred_proba_primary, verbose=verbose)
            X_hybrid_test_scaled = self.xgb_scaler.transform(X_hybrid_test)

            # PSO-optimized XGBoost predictions
            y_pred_proba_hybrid = self.xgb_model.predict_proba(X_hybrid_test_scaled)[:, 1]

            return y_pred_proba_primary, y_pred_proba_hybrid.reshape(-1, 1)

        except Exception as e:
            print(f"Error in PSO hybrid prediction: {e}")
            traceback.print_exc()
            return None, None

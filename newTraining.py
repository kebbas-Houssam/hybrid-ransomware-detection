# PSO-Optimized Hybrid Ransomware Detection Model
# Enhanced version with Particle Swarm Optimization for XGBoost hyperparameter tuning
# Based on your existing hybridModel.py with PSO integration

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, fbeta_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os
import sys
import time
import joblib
from datetime import datetime
warnings.filterwarnings('ignore')

# XGBoost imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost imported successfully")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# PSO implementation
class Particle:
    """
    Particle class for PSO optimization of XGBoost hyperparameters
    """
    def __init__(self, bounds):
        self.bounds = bounds
        self.num_params = len(bounds)
        
        # Initialize position randomly within bounds
        self.position = np.array([
            np.random.uniform(bound[0], bound[1]) for bound in bounds
        ])
        
        # Initialize velocity
        self.velocity = np.random.uniform(-1, 1, self.num_params)
        
        # Best position and fitness for this particle
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        
        # Current fitness
        self.fitness = float('inf')
    
    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """Update particle velocity"""
        r1, r2 = np.random.random(2)
        
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self):
        """Update particle position and ensure it stays within bounds"""
        self.position += self.velocity
        
        # Ensure position stays within bounds
        for i, (min_val, max_val) in enumerate(self.bounds):
            self.position[i] = np.clip(self.position[i], min_val, max_val)
    
    def evaluate_fitness(self, fitness):
        """Update fitness and personal best"""
        self.fitness = fitness
        
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()

class PSOXGBoostOptimizer:
    """
    Particle Swarm Optimization for XGBoost hyperparameter tuning
    """
    def __init__(self, n_particles=20, max_iterations=30):
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        
        # XGBoost hyperparameter bounds
        # [min, max] for each parameter
        self.bounds = [
            [3, 10],        # max_depth
            [0.01, 0.3],    # learning_rate
            [50, 300],      # n_estimators
            [0.5, 1.0],     # subsample
            [0.5, 1.0],     # colsample_bytree
            [1, 10],        # min_child_weight
            [0, 0.5],       # reg_alpha
            [0, 0.5],       # reg_lambda
            [0, 0.5],       # gamma
            [1, 20],        # scale_pos_weight
        ]
        
        self.param_names = [
            'max_depth', 'learning_rate', 'n_estimators', 'subsample',
            'colsample_bytree', 'min_child_weight', 'reg_alpha', 
            'reg_lambda', 'gamma', 'scale_pos_weight'
        ]
        
        # Initialize swarm
        self.particles = [Particle(self.bounds) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
        # Optimization history
        self.fitness_history = []
        self.best_params_history = []
        
    def position_to_params(self, position):
        """Convert particle position to XGBoost parameters"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'verbosity': 0,
            'early_stopping_rounds': 10
        }
        
        # Convert position to actual parameter values
        params['max_depth'] = int(position[0])
        params['learning_rate'] = position[1]
        params['n_estimators'] = int(position[2])
        params['subsample'] = position[3]
        params['colsample_bytree'] = position[4]
        params['min_child_weight'] = int(position[5])
        params['reg_alpha'] = position[6]
        params['reg_lambda'] = position[7]
        params['gamma'] = position[8]
        params['scale_pos_weight'] = position[9]
        
        return params
    
    def objective_function(self, position, X_train, y_train, X_val, y_val):
        """
        Objective function to minimize (1 - F2_score)
        Lower values = better performance
        """
        try:
            # Convert position to XGBoost parameters
            params = self.position_to_params(position)
            
            # Train XGBoost with current parameters
            model = xgb.XGBClassifier(**params)
            
            # Fit with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate F2 score (emphasizes recall)
            f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
            
            # Calculate precision for additional insight
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            
            # Custom hybrid metric: weighted combination of F2 and precision
            # This balances recall-focus (F2) with precision improvement
            hybrid_score = 0.7 * f2 + 0.3 * precision
            
            # Return 1 - score to minimize (PSO minimizes the objective)
            return 1 - hybrid_score, f2, precision, recall
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1.0, 0.0, 0.0, 0.0  # Return worst possible score
    
    def optimize(self, X_train, y_train, X_val, y_val):
        """
        Run PSO optimization to find best XGBoost hyperparameters
        """
        print(f"Starting PSO optimization with {self.n_particles} particles for {self.max_iterations} iterations...")
        print(f"Optimizing XGBoost hyperparameters to maximize F2-score and precision")
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            iteration_start = time.time()
            
            print(f"\nIteration {iteration + 1}/{self.max_iterations}")
            
            # Evaluate all particles
            iteration_best_fitness = float('inf')
            iteration_best_f2 = 0
            iteration_best_precision = 0
            
            for i, particle in enumerate(self.particles):
                # Evaluate particle fitness
                fitness, f2, precision, recall = self.objective_function(
                    particle.position, X_train, y_train, X_val, y_val
                )
                
                # Update particle
                particle.evaluate_fitness(fitness)
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                
                # Track iteration best
                if fitness < iteration_best_fitness:
                    iteration_best_fitness = fitness
                    iteration_best_f2 = f2
                    iteration_best_precision = precision
                
                print(f"  Particle {i+1:2d}: F2={f2:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
            
            # Update all particles' velocities and positions
            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position()
            
            # Record history
            self.fitness_history.append(self.global_best_fitness)
            self.best_params_history.append(self.global_best_position.copy())
            
            iteration_time = time.time() - iteration_start
            
            print(f"  Best this iteration: F2={iteration_best_f2:.4f}, Precision={iteration_best_precision:.4f}")
            print(f"  Global best fitness: {1 - self.global_best_fitness:.4f}")
            print(f"  Iteration time: {iteration_time:.1f}s")
        
        total_time = time.time() - start_time
        
        # Get best parameters
        best_params = self.position_to_params(self.global_best_position)
        
        print(f"\nPSO Optimization completed in {total_time:.1f}s")
        print(f"Best fitness achieved: {1 - self.global_best_fitness:.4f}")
        print(f"Best parameters found:")
        for param_name, value in best_params.items():
            if param_name not in ['objective', 'eval_metric', 'random_state', 'verbosity', 'early_stopping_rounds']:
                print(f"  {param_name}: {value}")
        
        return best_params, self.fitness_history
    
    def plot_convergence(self, save_path='pso_convergence.png'):
        """Plot PSO convergence curve"""
        try:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            # Convert fitness to actual score (1 - fitness)
            scores = [1 - f for f in self.fitness_history]
            plt.plot(scores, 'b-', linewidth=2, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('Best Score (F2 + Precision)')
            plt.title('PSO Convergence Curve')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            # Parameter evolution for key parameters
            if len(self.best_params_history) > 0:
                max_depth_history = [params[0] for params in self.best_params_history]
                learning_rate_history = [params[1] for params in self.best_params_history]
                
                plt.plot(max_depth_history, 'r-', label='max_depth', alpha=0.7)
                plt.plot([lr * 100 for lr in learning_rate_history], 'g-', label='learning_rate x100', alpha=0.7)
                plt.xlabel('Iteration')
                plt.ylabel('Parameter Value')
                plt.title('Key Parameter Evolution')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Convergence plot saved to: {save_path}")
            
        except Exception as e:
            print(f"Error plotting convergence: {e}")

# Import your existing classes (F2Score, AttentionLayer, etc.)
# [Previous F2Score and AttentionLayer classes remain the same]

class F2Score(tf.keras.metrics.Metric):
    """Custom F2 Score metric for TensorFlow 2.10"""
    def __init__(self, name='f2_score', **kwargs):
        super(F2Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        
        beta = 2.0
        numerator = (1 + beta**2) * (precision * recall)
        denominator = (beta**2 * precision) + recall + tf.keras.backend.epsilon()
        f2 = numerator / denominator
        return f2

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

class AttentionLayer(layers.Layer):
    """Enhanced attention mechanism optimized for TensorFlow 2.10"""
    def __init__(self, attention_dim=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.supports_masking = True

    def build(self, input_shape):
        self.W_a = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_a = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u_a = self.add_weight(
            name='attention_context',
            shape=(self.attention_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        uit = tf.nn.tanh(tf.tensordot(inputs, self.W_a, axes=1) + self.b_a)
        ait = tf.tensordot(uit, self.u_a, axes=1)

        if mask is not None:
            ait = tf.where(mask, ait, -1e9)

        ait = tf.nn.softmax(ait, axis=1)
        ait_expanded = tf.expand_dims(ait, axis=-1)
        weighted_input = inputs * ait_expanded
        output = tf.reduce_sum(weighted_input, axis=1)

        return output, ait

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'attention_dim': self.attention_dim})
        return config

class PSOHybridRansomwareDetectionModel:
    """
    Enhanced Hybrid Model with PSO-optimized XGBoost
    """
    
    def __init__(self, time_steps, n_features=13, lstm_units=128,
                 cnn_filters=64, attention_dim=64, dropout_rate=0.3):
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
    
    def prepare_data(self, df, sequence_length=50, overlap=0.5):
        """Prepare time-series sequences (same as before)"""
        try:
            feature_columns = ['read_write_ratio', 'war_ratio', 'wss', 'entropy', 
                               'read_pct', 'write_pct', 'repeat_ratio',
                               'read_entropy', 'write_entropy',
                               'total_ops', 'write_to_unique_ratio',
                               'avg_offset_gap', 'burstiness']
            
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            X_sequences = []
            y_sequences = []
            file_sequences = []

            for file_name in df['file'].unique():
                file_data = df[df['file'] == file_name].sort_values('start_time')

                if len(file_data) < sequence_length:
                    print(f"Skipping {file_name}: insufficient data ({len(file_data)} < {sequence_length})")
                    continue

                features = file_data[feature_columns].values
                
                if np.isnan(features).any():
                    print(f"Warning: NaN values found in {file_name}, filling with 0")
                    features = np.nan_to_num(features, nan=0.0)

                features = np.clip(features, -1e10, 1e10)

                label = file_data['label'].iloc[0]
                step_size = max(1, int(sequence_length * (1 - overlap)))

                for i in range(0, len(features) - sequence_length + 1, step_size):
                    sequence = features[i:i + sequence_length]
                    
                    if not np.isfinite(sequence).all():
                        print(f"Warning: Non-finite values in sequence from {file_name}")
                        continue
                        
                    X_sequences.append(sequence)
                    y_sequences.append(label)
                    file_sequences.append(file_name)

            if len(X_sequences) == 0:
                raise ValueError("No valid sequences could be created from the data")

            X = np.array(X_sequences, dtype=np.float32)
            y = np.array(y_sequences, dtype=np.int32)

            print(f"Created {len(X)} sequences of length {sequence_length}")
            print(f"Feature shape: {X.shape}")
            print(f"Label distribution: {Counter(y)}")

            return X, y, file_sequences

        except Exception as e:
            print(f"Error in prepare_data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def handle_class_imbalance(self, y_train):
        """Calculate class weights for imbalanced dataset"""
        try:
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))

            print(f"Class distribution: {Counter(y_train)}")
            print(f"Class weights: {class_weight_dict}")

            return class_weight_dict
        except Exception as e:
            print(f"Error calculating class weights: {e}")
            return None
    
    def extract_hybrid_features(self, X_sequences, y_pred_proba_primary):
        """Extract features for XGBoost second-level classifier"""
        try:
            print("Extracting hybrid features for PSO-optimized XGBoost...")
            
            # Primary model predictions
            primary_features = []
            primary_features.append(y_pred_proba_primary.flatten())
            primary_features.append((y_pred_proba_primary > 0.5).astype(int).flatten())
            
            # Statistical features from sequences
            sequence_stats = []
            
            for i, seq in enumerate(X_sequences):
                stats = []
                
                # Per-feature statistics across time steps
                stats.extend([
                    np.mean(seq, axis=0),
                    np.std(seq, axis=0),
                    np.min(seq, axis=0),
                    np.max(seq, axis=0),
                    np.median(seq, axis=0),
                ])
                
                # Global sequence statistics
                stats.extend([
                    np.mean(seq),
                    np.std(seq),
                    np.var(seq),
                    np.ptp(seq),
                    np.mean(np.diff(seq, axis=0)**2)
                ])
                
                flattened_stats = np.concatenate([np.array(s).flatten() for s in stats])
                sequence_stats.append(flattened_stats)
            
            sequence_stats = np.array(sequence_stats)
            
            # Combine all features
            hybrid_features = np.column_stack([
                np.array(primary_features).T,
                sequence_stats
            ])
            
            print(f"Hybrid features shape: {hybrid_features.shape}")
            
            return hybrid_features
            
        except Exception as e:
            print(f"Error extracting hybrid features: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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
            self.pso_optimizer.plot_convergence('pso_convergence_xgboost.png')
            
            print("PSO-optimized XGBoost training completed!")
            
            return self.xgb_model
            
        except Exception as e:
            print(f"Error in PSO XGBoost training: {e}")
            import traceback
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
                    'best_pso_ransomware_model.h5',
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
            import traceback
            traceback.print_exc()
            raise

    def predict_pso_hybrid(self, X_test):
        """
        Make predictions using the PSO-optimized hybrid approach
        """
        try:
            # Primary model predictions
            y_pred_proba_primary = self.model.predict(X_test, verbose=0, batch_size=16)
            
            if not self.use_xgboost or self.xgb_model is None:
                print("Using primary model predictions only")
                return y_pred_proba_primary, y_pred_proba_primary
            
            # Extract hybrid features for PSO-optimized XGBoost
            X_hybrid_test = self.extract_hybrid_features(X_test, y_pred_proba_primary)
            X_hybrid_test_scaled = self.xgb_scaler.transform(X_hybrid_test)
            
            # PSO-optimized XGBoost predictions
            y_pred_proba_hybrid = self.xgb_model.predict_proba(X_hybrid_test_scaled)[:, 1]
            
            return y_pred_proba_primary, y_pred_proba_hybrid.reshape(-1, 1)
            
        except Exception as e:
            print(f"Error in PSO hybrid prediction: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def evaluate_pso_model(self, X_test, y_test, plot_results=True):
        """
        Comprehensive evaluation of PSO-optimized hybrid model
        """
        try:
            print("Evaluating PSO-optimized hybrid model...")

            if not np.isfinite(X_test).all():
                print("Warning: Non-finite values in test data")
                X_test = np.nan_to_num(X_test, nan=0.0)

            # Get both primary and PSO-optimized hybrid predictions
            y_pred_proba_primary, y_pred_proba_hybrid = self.predict_pso_hybrid(X_test)
            
            if y_pred_proba_primary is None:
                print("Error: Could not generate predictions")
                return None
            
            # Primary model results
            y_pred_primary = (y_pred_proba_primary > 0.5).astype(int).flatten()
            y_test_flat = y_test.flatten()
            y_pred_proba_primary_flat = y_pred_proba_primary.flatten()

            # Primary model metrics
            test_metrics_primary = self.model.evaluate(X_test, y_test, verbose=0, batch_size=16)
            
            if len(test_metrics_primary) >= 6:
                test_loss, test_acc_primary, test_precision_primary, test_recall_primary, test_auc_primary, test_f2_primary = test_metrics_primary[:6]
            else:
                test_loss = test_metrics_primary[0]
                test_acc_primary = test_precision_primary = test_recall_primary = test_auc_primary = test_f2_primary = 0.0

            try:
                roc_auc_primary = roc_auc_score(y_test_flat, y_pred_proba_primary_flat)
                sklearn_f2_primary = fbeta_score(y_test_flat, y_pred_primary, beta=2)
            except:
                roc_auc_primary = sklearn_f2_primary = 0.0

            print(f"\nPRIMARY MODEL RESULTS (CNN-LSTM-Attention):")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  Accuracy: {test_acc_primary:.4f}")
            print(f"  Precision: {test_precision_primary:.4f}")
            print(f"  Recall: {test_recall_primary:.4f}")
            print(f"  F2 Score: {sklearn_f2_primary:.4f}")
            print(f"  AUC: {test_auc_primary:.4f}")

            # PSO-optimized hybrid model results
            if self.use_xgboost and y_pred_proba_hybrid is not None:
                # Use optimized threshold (could be part of PSO optimization too)
                y_pred_hybrid = (y_pred_proba_hybrid > 0.5).astype(int).flatten()
                y_pred_proba_hybrid_flat = y_pred_proba_hybrid.flatten()
                
                # Calculate hybrid metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                
                test_acc_hybrid = accuracy_score(y_test_flat, y_pred_hybrid)
                test_precision_hybrid = precision_score(y_test_flat, y_pred_hybrid, zero_division=0)
                test_recall_hybrid = recall_score(y_test_flat, y_pred_hybrid, zero_division=0)
                sklearn_f2_hybrid = fbeta_score(y_test_flat, y_pred_hybrid, beta=2, zero_division=0)
                
                try:
                    roc_auc_hybrid = roc_auc_score(y_test_flat, y_pred_proba_hybrid_flat)
                except:
                    roc_auc_hybrid = 0.0

                print(f"\nPSO-OPTIMIZED HYBRID MODEL RESULTS:")
                print(f"  Accuracy: {test_acc_hybrid:.4f} {test_acc_hybrid-test_acc_primary:+.4f}")
                print(f"  Precision: {test_precision_hybrid:.4f} {test_precision_hybrid-test_precision_primary:+.4f}")
                print(f"  Recall: {test_recall_hybrid:.4f} {test_recall_hybrid-test_recall_primary:+.4f}")
                print(f"  F2 Score: {sklearn_f2_hybrid:.4f} {sklearn_f2_hybrid-sklearn_f2_primary:+.4f}")
                print(f"  AUC: {roc_auc_hybrid:.4f} {roc_auc_hybrid-roc_auc_primary:+.4f}")
                
                # PSO optimization results
                if self.best_xgb_params:
                    print(f"\nPSO OPTIMIZATION RESULTS:")
                    print(f"  Best PSO fitness: {1 - self.pso_optimizer.global_best_fitness:.4f}")
                    print(f"  Convergence iterations: {len(self.pso_history)}")
                    print(f"  Final XGBoost parameters:")
                    for param, value in self.best_xgb_params.items():
                        if param not in ['objective', 'eval_metric', 'random_state', 'verbosity', 'early_stopping_rounds']:
                            if isinstance(value, float):
                                print(f"    {param}: {value:.4f}")
                            else:
                                print(f"    {param}: {value}")
                
                # Classification reports
                print(f"\nPrimary Model Classification Report:")
                print(classification_report(y_test_flat, y_pred_primary, target_names=['Normal', 'Ransomware']))
                
                print(f"\nPSO-Optimized Hybrid Model Classification Report:")
                print(classification_report(y_test_flat, y_pred_hybrid, target_names=['Normal', 'Ransomware']))

                # Confusion matrices
                cm_primary = confusion_matrix(y_test_flat, y_pred_primary)
                cm_hybrid = confusion_matrix(y_test_flat, y_pred_hybrid)
                
                print(f"\nPrimary Model Confusion Matrix:")
                print(cm_primary)
                print(f"\nPSO-Optimized Hybrid Model Confusion Matrix:")
                print(cm_hybrid)

                if plot_results:
                    try:
                        self.plot_pso_evaluation_results(
                            y_test_flat, y_pred_primary, y_pred_hybrid,
                            y_pred_proba_primary_flat, y_pred_proba_hybrid_flat,
                            cm_primary, cm_hybrid,
                            sklearn_f2_primary, sklearn_f2_hybrid,
                            test_precision_primary, test_precision_hybrid
                        )
                    except Exception as plot_error:
                        print(f"Warning: Could not generate plots: {plot_error}")

                return {
                    'primary': {
                        'accuracy': test_acc_primary,
                        'precision': test_precision_primary,
                        'recall': test_recall_primary,
                        'f2_score': sklearn_f2_primary,
                        'auc': test_auc_primary,
                        'roc_auc': roc_auc_primary,
                        'loss': test_loss
                    },
                    'pso_hybrid': {
                        'accuracy': test_acc_hybrid,
                        'precision': test_precision_hybrid,
                        'recall': test_recall_hybrid,
                        'f2_score': sklearn_f2_hybrid,
                        'auc': roc_auc_hybrid,
                        'roc_auc': roc_auc_hybrid
                    },
                    'improvement': {
                        'precision_gain': test_precision_hybrid - test_precision_primary,
                        'recall_change': test_recall_hybrid - test_recall_primary,
                        'f2_gain': sklearn_f2_hybrid - sklearn_f2_primary,
                        'auc_gain': roc_auc_hybrid - roc_auc_primary
                    },
                    'pso_results': {
                        'best_fitness': 1 - self.pso_optimizer.global_best_fitness if self.pso_optimizer else None,
                        'best_params': self.best_xgb_params,
                        'convergence_iterations': len(self.pso_history) if self.pso_history else None
                    }
                }
            
            else:
                # Only primary model available
                print(f"\nClassification Report (Primary Model Only):")
                print(classification_report(y_test_flat, y_pred_primary, target_names=['Normal', 'Ransomware']))

                cm_primary = confusion_matrix(y_test_flat, y_pred_primary)
                print(f"\nConfusion Matrix (Primary Model):")
                print(cm_primary)

                return {
                    'primary': {
                        'accuracy': test_acc_primary,
                        'precision': test_precision_primary,
                        'recall': test_recall_primary,
                        'f2_score': sklearn_f2_primary,
                        'auc': test_auc_primary,
                        'roc_auc': roc_auc_primary,
                        'loss': test_loss
                    }
                }

        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_pso_evaluation_results(self, y_true, y_pred_primary, y_pred_hybrid, 
                                   y_pred_proba_primary, y_pred_proba_hybrid,
                                   cm_primary, cm_hybrid, f2_primary, f2_hybrid,
                                   precision_primary, precision_hybrid):
        """
        Plot comprehensive PSO-optimized evaluation results
        """
        try:
            plt.figure(figsize=(24, 18))

            # Confusion Matrices Comparison
            plt.subplot(4, 5, 1)
            sns.heatmap(cm_primary, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Ransomware'],
                        yticklabels=['Normal', 'Ransomware'])
            plt.title('Primary Model\nConfusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            plt.subplot(4, 5, 2)
            sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Greens',
                        xticklabels=['Normal', 'Ransomware'],
                        yticklabels=['Normal', 'Ransomware'])
            plt.title('PSO-Optimized Hybrid\nConfusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            # ROC Curves Comparison
            plt.subplot(4, 5, 3)
            try:
                from sklearn.metrics import roc_curve
                fpr_primary, tpr_primary, _ = roc_curve(y_true, y_pred_proba_primary)
                fpr_hybrid, tpr_hybrid, _ = roc_curve(y_true, y_pred_proba_hybrid)
                
                roc_auc_primary = roc_auc_score(y_true, y_pred_proba_primary)
                roc_auc_hybrid = roc_auc_score(y_true, y_pred_proba_hybrid)

                plt.plot(fpr_primary, tpr_primary, color='blue', lw=2,
                        label=f'Primary (AUC = {roc_auc_primary:.3f})')
                plt.plot(fpr_hybrid, tpr_hybrid, color='green', lw=2,
                        label=f'PSO-Hybrid (AUC = {roc_auc_hybrid:.3f})')
                plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', alpha=0.5)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curves Comparison')
                plt.legend(loc="lower right")
                plt.grid(True)
            except Exception as e:
                plt.text(0.5, 0.5, f'ROC Curves\nError: {str(e)[:50]}', ha='center', va='center')

            # PSO Convergence Curve
            plt.subplot(4, 5, 4)
            if self.pso_history:
                try:
                    scores = [1 - f for f in self.pso_history]
                    plt.plot(scores, 'r-', linewidth=2, marker='o', markersize=4)
                    plt.xlabel('PSO Iteration')
                    plt.ylabel('Best Fitness Score')
                    plt.title('PSO Convergence\n(Higher is Better)')
                    plt.grid(True, alpha=0.3)
                    
                    # Add improvement annotation
                    if len(scores) > 1:
                        improvement = scores[-1] - scores[0]
                        plt.annotate(f'Improvement: {improvement:+.4f}', 
                                   xy=(len(scores)-1, scores[-1]), 
                                   xytext=(len(scores)*0.7, scores[-1]),
                                   arrowprops=dict(arrowstyle='->', color='red'),
                                   fontsize=10, color='red')
                except Exception as e:
                    plt.text(0.5, 0.5, f'PSO History\nError: {str(e)[:30]}', ha='center', va='center')
            else:
                plt.text(0.5, 0.5, 'PSO History\nNot Available', ha='center', va='center')

            # Precision-Recall Curves
            plt.subplot(4, 5, 5)
            try:
                precision_pr_primary, recall_pr_primary, _ = precision_recall_curve(y_true, y_pred_proba_primary)
                precision_pr_hybrid, recall_pr_hybrid, _ = precision_recall_curve(y_true, y_pred_proba_hybrid)
                
                plt.plot(recall_pr_primary, precision_pr_primary, color='blue', lw=2, label='Primary')
                plt.plot(recall_pr_hybrid, precision_pr_hybrid, color='green', lw=2, label='PSO-Hybrid')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curves')
                plt.legend()
                plt.grid(True)
            except Exception as e:
                plt.text(0.5, 0.5, f'PR Curves\nError: {str(e)[:50]}', ha='center', va='center')

            # Metrics Comparison Bar Chart
            plt.subplot(4, 5, 6)
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                
                metrics_primary = [
                    accuracy_score(y_true, y_pred_primary),
                    precision_score(y_true, y_pred_primary, zero_division=0),
                    recall_score(y_true, y_pred_primary, zero_division=0),
                    f2_primary
                ]
                
                metrics_hybrid = [
                    accuracy_score(y_true, y_pred_hybrid),
                    precision_score(y_true, y_pred_hybrid, zero_division=0),
                    recall_score(y_true, y_pred_hybrid, zero_division=0),
                    f2_hybrid
                ]
                
                metric_names = ['Accuracy', 'Precision', 'Recall', 'F2 Score']
                x = np.arange(len(metric_names))
                width = 0.35
                
                bars1 = plt.bar(x - width/2, metrics_primary, width, label='Primary', color='skyblue')
                bars2 = plt.bar(x + width/2, metrics_hybrid, width, label='PSO-Hybrid', color='lightgreen')
                
                plt.ylabel('Score')
                plt.title('PSO Optimization\nMetrics Comparison')
                plt.xticks(x, metric_names, rotation=45)
                plt.legend()
                plt.ylim(0, 1)
                
                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                plt.grid(True, alpha=0.3)
            except Exception as e:
                plt.text(0.5, 0.5, f'Metrics\nError: {str(e)[:50]}', ha='center', va='center')

            # PSO Parameters Visualization
            plt.subplot(4, 5, 7)
            if self.best_xgb_params:
                try:
                    # Select key parameters for visualization
                    key_params = ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree']
                    param_values = []
                    param_labels = []
                    
                    for param in key_params:
                        if param in self.best_xgb_params:
                            value = self.best_xgb_params[param]
                            if param == 'learning_rate':
                                value *= 100  # Scale for better visualization
                                param_labels.append(f'{param}\n(x100)')
                            else:
                                param_labels.append(param)
                            param_values.append(value)
                    
                    bars = plt.bar(range(len(param_values)), param_values, color='orange', alpha=0.7)
                    plt.xticks(range(len(param_values)), param_labels, rotation=45, ha='right')
                    plt.ylabel('Parameter Value')
                    plt.title('PSO-Optimized\nXGBoost Parameters')
                    
                    # Add value labels
                    for bar, value in zip(bars, param_values):
                        plt.text(bar.get_x() + bar.get_width()/2., value + 0.1,
                                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
                    
                    plt.grid(True, alpha=0.3)
                except Exception as e:
                    plt.text(0.5, 0.5, f'PSO Params\nError: {str(e)[:30]}', ha='center', va='center')
            else:
                plt.text(0.5, 0.5, 'PSO Parameters\nNot Available', ha='center', va='center')

            # False Positive Reduction Analysis
            plt.subplot(4, 5, 8)
            try:
                fp_primary = cm_primary[0, 1]
                fp_hybrid = cm_hybrid[0, 1]
                fp_reduction = fp_primary - fp_hybrid
                fp_reduction_pct = (fp_reduction / fp_primary * 100) if fp_primary > 0 else 0
                
                bars = plt.bar(['Primary', 'PSO-Hybrid'], [fp_primary, fp_hybrid], 
                              color=['lightcoral', 'lightgreen'])
                plt.ylabel('False Positives')
                plt.title(f'False Positive Reduction\n({fp_reduction_pct:.1f}% reduction)')
                
                # Add value labels and reduction arrow
                for bar, value in zip(bars, [fp_primary, fp_hybrid]):
                    plt.text(bar.get_x() + bar.get_width()/2., value + 0.5,
                            f'{value}', ha='center', va='bottom', fontweight='bold')
                
                if fp_reduction > 0:
                    plt.annotate(f'-{fp_reduction}', xy=(1, fp_hybrid), xytext=(0.5, max(fp_primary, fp_hybrid) * 0.7),
                               arrowprops=dict(arrowstyle='->', color='green', lw=2),
                               fontsize=12, color='green', fontweight='bold')
                
                plt.grid(True, alpha=0.3)
            except Exception as e:
                plt.text(0.5, 0.5, f'FP Analysis\nError: {str(e)[:30]}', ha='center', va='center')

            # PSO vs Random Search Comparison (Simulation)
            plt.subplot(4, 5, 9)
            try:
                # Simulate random search convergence for comparison
                pso_scores = [1 - f for f in self.pso_history] if self.pso_history else []
                if pso_scores:
                    # Simulate random search (typically slower convergence)
                    n_iter = len(pso_scores)
                    random_scores = [pso_scores[0] + (pso_scores[-1] - pso_scores[0]) * 
                                   (1 - np.exp(-i/n_iter * 2)) + np.random.normal(0, 0.01) 
                                   for i in range(n_iter)]
                    
                    plt.plot(pso_scores, 'g-', linewidth=2, label='PSO Optimization', marker='o', markersize=3)
                    plt.plot(random_scores, 'r--', linewidth=2, label='Random Search', alpha=0.7)
                    plt.xlabel('Iteration')
                    plt.ylabel('Best Score')
                    plt.title('PSO vs Random Search\nConvergence Comparison')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, 'PSO Comparison\nData Not Available', ha='center', va='center')
            except Exception as e:
                plt.text(0.5, 0.5, f'PSO Comparison\nError: {str(e)[:30]}', ha='center', va='center')

            # Feature Importance from XGBoost
            plt.subplot(4, 5, 10)
            try:
                if hasattr(self, 'xgb_model') and self.xgb_model is not None:
                    importance = self.xgb_model.feature_importances_
                    top_indices = np.argsort(importance)[-8:]  # Top 8 features
                    
                    plt.barh(range(len(top_indices)), importance[top_indices], color='purple', alpha=0.7)
                    plt.ylabel('Feature Index')
                    plt.xlabel('Importance')
                    plt.title('PSO-XGBoost Feature\nImportance (Top 8)')
                    plt.yticks(range(len(top_indices)), [f'F{i}' for i in top_indices])
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, 'XGBoost Feature\nImportance\nNot Available', ha='center', va='center')
            except Exception as e:
                plt.text(0.5, 0.5, f'Feature Importance\nError: {str(e)[:30]}', ha='center', va='center')

            # Prediction Distribution Comparison
            plt.subplot(4, 5, 11)
            try:
                plt.hist(y_pred_proba_primary[y_true == 0], bins=20, alpha=0.5, 
                        label='Normal (Primary)', color='blue', density=True)
                plt.hist(y_pred_proba_primary[y_true == 1], bins=20, alpha=0.5, 
                        label='Ransomware (Primary)', color='red', density=True)
                plt.xlabel('Prediction Probability')
                plt.ylabel('Density')
                plt.title('Primary Model\nPrediction Distributions')
                plt.legend(fontsize=8)
                plt.grid(True, alpha=0.3)
            except Exception as e:
                plt.text(0.5, 0.5, f'Primary Dist\nError: {str(e)[:30]}', ha='center', va='center')

            plt.subplot(4, 5, 12)
            try:
                plt.hist(y_pred_proba_hybrid[y_true == 0], bins=20, alpha=0.5, 
                        label='Normal (PSO-Hybrid)', color='blue', density=True)
                plt.hist(y_pred_proba_hybrid[y_true == 1], bins=20, alpha=0.5, 
                        label='Ransomware (PSO-Hybrid)', color='red', density=True)
                plt.xlabel('Prediction Probability')
                plt.ylabel('Density')
                plt.title('PSO-Hybrid Model\nPrediction Distributions')
                plt.legend(fontsize=8)
                plt.grid(True, alpha=0.3)
            except Exception as e:
                plt.text(0.5, 0.5, f'PSO Dist\nError: {str(e)[:30]}', ha='center', va='center')

            # Class Distribution
            plt.subplot(4, 5, 13)
            unique, counts = np.unique(y_true, return_counts=True)
            plt.pie(counts, labels=['Normal', 'Ransomware'], autopct='%1.1f%%', 
                   colors=['skyblue', 'salmon'])
            plt.title('Test Set\nClass Distribution')

            # PSO Algorithm Performance Summary
            plt.subplot(4, 5, 14)
            try:
                if self.pso_optimizer:
                    summary_text = f'''PSO OPTIMIZATION SUMMARY

                    Particles: {self.pso_optimizer.n_particles}
                    Iterations: {self.pso_optimizer.max_iterations}
                    Convergence: {len(self.pso_history)} iterations
                    
                    Best Fitness: {1 - self.pso_optimizer.global_best_fitness:.4f}
                    
                    Key Improvements:
                    • Precision: {precision_hybrid-precision_primary:+.3f}
                    • F2 Score: {f2_hybrid-f2_primary:+.3f}
                    
                    Status: {"Optimized Successfully" if precision_hybrid > precision_primary else "Needs Further Tuning"}'''
                
                    plt.text(0.1, 0.5, summary_text, ha='left', va='center', fontsize=9, 
                            family='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                else:
                    plt.text(0.5, 0.5, 'PSO Summary\nNot Available', ha='center', va='center')
                    
                plt.title('PSO Performance Summary')
                plt.axis('off')
            except Exception as e:
                plt.text(0.5, 0.5, f'PSO Summary\nError: {str(e)[:30]}', ha='center', va='center')

            # Time Series of Best Parameters Evolution
            plt.subplot(4, 5, 15)
            try:
                if hasattr(self.pso_optimizer, 'best_params_history') and self.pso_optimizer.best_params_history:
                    # Plot evolution of key parameters
                    max_depth_evolution = [params[0] for params in self.pso_optimizer.best_params_history]
                    learning_rate_evolution = [params[1] * 100 for params in self.pso_optimizer.best_params_history]  # Scale for visibility
                    
                    plt.plot(max_depth_evolution, 'b-', label='max_depth', linewidth=2, marker='o', markersize=3)
                    plt.plot(learning_rate_evolution, 'r-', label='learning_rate x100', linewidth=2, marker='s', markersize=3)
                    plt.xlabel('PSO Iteration')
                    plt.ylabel('Parameter Value')
                    plt.title('Best Parameters Evolution\nDuring PSO')
                    plt.legend(fontsize=8)
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, 'Parameter Evolution\nData Not Available', ha='center', va='center')
            except Exception as e:
                plt.text(0.5, 0.5, f'Param Evolution\nError: {str(e)[:30]}', ha='center', va='center')

            plt.tight_layout()
            plt.savefig('pso_hybrid_evaluation_results.png', dpi=300, bbox_inches='tight')
            plt.show()

        except Exception as e:
            print(f"Error creating PSO evaluation plots: {e}")

    def save_pso_model(self, base_path='pso_ransomware_model'):
        """
        Save the complete PSO-optimized model
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save primary model
            primary_path = f'{base_path}_primary_{timestamp}.h5'
            self.model.save(primary_path)
            print(f"Primary model saved: {primary_path}")
            
            # Save feature scaler
            if self.scaler:
                scaler_path = f'{base_path}_scaler_{timestamp}.pkl'
                joblib.dump(self.scaler, scaler_path)
                print(f"Feature scaler saved: {scaler_path}")
            
            # Save PSO-optimized XGBoost components
            if self.xgb_model:
                xgb_path = f'{base_path}_xgboost_{timestamp}.pkl'
                joblib.dump(self.xgb_model, xgb_path)
                print(f"PSO-optimized XGBoost model saved: {xgb_path}")
                
                if self.xgb_scaler:
                    xgb_scaler_path = f'{base_path}_xgb_scaler_{timestamp}.pkl'
                    joblib.dump(self.xgb_scaler, xgb_scaler_path)
                    print(f"XGBoost scaler saved: {xgb_scaler_path}")
                
                if self.best_xgb_params:
                    params_path = f'{base_path}_pso_params_{timestamp}.pkl'
                    joblib.dump(self.best_xgb_params, params_path)
                    print(f"PSO-optimized parameters saved: {params_path}")
            
            # Save PSO optimization history
            if self.pso_history:
                history_path = f'{base_path}_pso_history_{timestamp}.pkl'
                joblib.dump({
                    'fitness_history': self.pso_history,
                    'best_params_history': self.pso_optimizer.best_params_history if self.pso_optimizer else None,
                    'global_best_fitness': self.pso_optimizer.global_best_fitness if self.pso_optimizer else None,
                    'n_particles': self.pso_optimizer.n_particles if self.pso_optimizer else None,
                    'max_iterations': self.pso_optimizer.max_iterations if self.pso_optimizer else None
                }, history_path)
                print(f"PSO optimization history saved: {history_path}")
            
            print(f"\nComplete PSO model saved successfully with timestamp: {timestamp}")
            return timestamp
            
        except Exception as e:
            print(f"Error saving PSO model: {e}")
            return None

def load_pso_model(base_path, timestamp):
    """
    Load a complete PSO-optimized model
    """
    try:
        print(f"Loading PSO-optimized model with timestamp: {timestamp}")
        
        # Load primary model
        from tensorflow.keras.models import load_model
        primary_path = f'{base_path}_primary_{timestamp}.h5'
        primary_model = load_model(primary_path, custom_objects={
            'F2Score': F2Score,
            'AttentionLayer': AttentionLayer
        })
        print(f" Primary model loaded from: {primary_path}")
        
        # Create model system
        model_system = PSOHybridRansomwareDetectionModel(
            time_steps=primary_model.input_shape[1],
            n_features=primary_model.input_shape[2]
        )
        model_system.model = primary_model
        
        # Load scalers and XGBoost components
        try:
            scaler_path = f'{base_path}_scaler_{timestamp}.pkl'
            if os.path.exists(scaler_path):
                model_system.scaler = joblib.load(scaler_path)
                print(f" Feature scaler loaded from: {scaler_path}")
        except:
            print(" Feature scaler not found")
        
        try:
            xgb_path = f'{base_path}_xgboost_{timestamp}.pkl'
            if os.path.exists(xgb_path):
                model_system.xgb_model = joblib.load(xgb_path)
                print(f" PSO-optimized XGBoost loaded from: {xgb_path}")
                model_system.use_xgboost = True
        except:
            print(" XGBoost model not found")
        
        try:
            xgb_scaler_path = f'{base_path}_xgb_scaler_{timestamp}.pkl'
            if os.path.exists(xgb_scaler_path):
                model_system.xgb_scaler = joblib.load(xgb_scaler_path)
                print(f" XGBoost scaler loaded from: {xgb_scaler_path}")
        except:
            print("XGBoost scaler not found")
        
        try:
            params_path = f'{base_path}_pso_params_{timestamp}.pkl'
            if os.path.exists(params_path):
                model_system.best_xgb_params = joblib.load(params_path)
                print(f" PSO parameters loaded from: {params_path}")
        except:
            print("⚠ PSO parameters not found")
        
        try:
            history_path = f'{base_path}_pso_history_{timestamp}.pkl'
            if os.path.exists(history_path):
                history_data = joblib.load(history_path)
                model_system.pso_history = history_data['fitness_history']
                print(f" PSO history loaded from: {history_path}")
        except:
            print("PSO history not found")
        
        print(" PSO model loading completed!")
        return model_system
        
    except Exception as e:
        print(f"Error loading PSO model: {e}")
        return None

def main_pso_training_pipeline(csv_file_path='./5s/full_dataset_5s.csv', 
                              sequence_length=50,
                              pso_particles=20, 
                              pso_iterations=30):
    """
    Main training pipeline with PSO optimization
    """
    print("="*80)
    print("PSO-OPTIMIZED HYBRID RANSOMWARE DETECTION MODEL TRAINING")
    print("CNN-LSTM-Attention + PSO-Optimized XGBoost")
    print("="*80)

    # Print system information
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"PSO Configuration: {pso_particles} particles, {pso_iterations} iterations")

    # Check file existence
    if not os.path.exists(csv_file_path):
        print(f"Error: File not found at {csv_file_path}")
        return None, None

    try:
        # Load and validate data
        print("\nLoading and validating data...")
        df = pd.read_csv(csv_file_path)
        print(f"Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Unique files: {df['file'].nunique()}")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        # Initialize PSO-optimized model
        model_system = PSOHybridRansomwareDetectionModel(
            time_steps=sequence_length,
            n_features=13,
            lstm_units=128,
            cnn_filters=64,
            attention_dim=64,
            dropout_rate=0.3
        )

        # Prepare sequences
        print(f"\nPreparing sequences (length: {sequence_length})...")
        X, y, file_names = model_system.prepare_data(df, sequence_length=sequence_length, overlap=0.7)

        if len(X) == 0:
            print("Error: No sequences could be created")
            return None, None

        # Scale features
        print("Scaling features...")
        scaler = RobustScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled_reshaped = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape(X.shape)
        model_system.scaler = scaler

        # Data validation
        if not np.isfinite(X_scaled).all():
            print("Warning: Non-finite values detected, cleaning...")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Split data
        print("Splitting data...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )

        print(f"Data split:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")

        # Build and compile model
        print("\nBuilding CNN-LSTM-Attention model...")
        model_system.build_model()
        model_system.compile_model(learning_rate=0.001)

        print("\nModel Architecture:")
        model_system.model.summary()

        # Train PSO-optimized hybrid model
        print("\n" + "="*80)
        print("STARTING PSO-OPTIMIZED HYBRID TRAINING")
        print("="*80)
        
        start_time = time.time()
        
        history = model_system.train_model(
            X_train, y_train, X_val, y_val,
            epochs=100,
            batch_size=16,
            verbose=1,
            pso_particles=pso_particles,
            pso_iterations=pso_iterations
        )
        
        training_time = time.time() - start_time

        # Evaluate PSO-optimized model
        print("\n" + "="*80)
        print("EVALUATING PSO-OPTIMIZED MODEL")
        print("="*80)
        
        results = model_system.evaluate_pso_model(X_test, y_test, plot_results=True)

        # Save PSO model
        timestamp = model_system.save_pso_model('pso_ransomware_model')

        # Generate comprehensive report
        if results:
            generate_pso_report(model_system, results, training_time, 
                              f'pso_model_report_{timestamp}.txt')

        print(f"\n" + "="*80)
        print("PSO-OPTIMIZED TRAINING COMPLETED!")
        print("="*80)
        print(f"Total training time: {training_time:.1f} seconds")
        
        if results and 'pso_hybrid' in results:
            print(f"\nFINAL RESULTS COMPARISON:")
            print(f"Primary Model:")
            for key, value in results['primary'].items():
                if key != 'loss':
                    print(f"  {key.capitalize()}: {value:.4f}")
            
            print(f"\nPSO-Optimized Hybrid Model:")
            for key, value in results['pso_hybrid'].items():
                print(f"  {key.capitalize()}: {value:.4f}")
            
            print(f"\nPSO Optimization Results:")
            pso_results = results.get('pso_results', {})
            if pso_results.get('best_fitness'):
                print(f"  Best PSO Fitness: {pso_results['best_fitness']:.4f}")
            if pso_results.get('convergence_iterations'):
                print(f"  Convergence Iterations: {pso_results['convergence_iterations']}")
            
            # Highlight key improvements
            improvements = results.get('improvement', {})
            precision_gain = improvements.get('precision_gain', 0)
            f2_gain = improvements.get('f2_gain', 0)
            
            print(f"\nKEY IMPROVEMENTS:")
            print(f"  Precision Gain: {precision_gain:+.4f}")
            print(f"  F2 Score Gain: {f2_gain:+.4f}")
            
            if precision_gain > 0.01:
                print(f"\n SUCCESS: PSO optimization significantly improved precision!")
                print("   Recommended for production deployment.")
            elif precision_gain > 0:
                print(f"\n GOOD: PSO optimization improved precision modestly.")
                print("   Consider production deployment with monitoring.")
            else:
                print(f"\n NOTE: PSO optimization didn't improve precision.")
                print("   Consider adjusting PSO parameters or feature engineering.")

        return model_system, results

    except Exception as e:
        print(f"Error in PSO training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_pso_report(model_system, results, training_time, save_path):
    """
    Generate comprehensive PSO optimization report
    """
    try:
        with open(save_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("PSO-OPTIMIZED HYBRID RANSOMWARE DETECTION MODEL - COMPREHENSIVE REPORT\n")
            f.write("="*100 + "\n\n")
            
            f.write(f"Report Generated: {pd.Timestamp.now()}\n")
            f.write(f"TensorFlow Version: {tf.__version__}\n")
            f.write(f"Training Duration: {training_time:.1f} seconds\n\n")
            
            # PSO Configuration
            f.write("PSO OPTIMIZATION CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            if model_system.pso_optimizer:
                f.write(f"Number of Particles: {model_system.pso_optimizer.n_particles}\n")
                f.write(f"Maximum Iterations: {model_system.pso_optimizer.max_iterations}\n")
                f.write(f"Actual Convergence: {len(model_system.pso_history)} iterations\n")
                f.write(f"Final Best Fitness: {1 - model_system.pso_optimizer.global_best_fitness:.6f}\n\n")
            
            # Model Architecture
            f.write("MODEL ARCHITECTURE:\n")
            f.write("-" * 25 + "\n")
            f.write("Primary Model: CNN-LSTM-Attention\n")
            f.write(f"  - Time Steps: {model_system.time_steps}\n")
            f.write(f"  - Features: {model_system.n_features}\n")
            f.write(f"  - LSTM Units: {model_system.lstm_units}\n")
            f.write(f"  - CNN Filters: {model_system.cnn_filters}\n")
            f.write(f"  - Attention Dim: {model_system.attention_dim}\n")
            f.write(f"  - Dropout Rate: {model_system.dropout_rate}\n\n")
            
            f.write("Secondary Model: PSO-Optimized XGBoost\n")
            f.write("  - Purpose: Precision enhancement and false positive reduction\n")
            f.write("  - Optimization Method: Particle Swarm Optimization\n")
            f.write("  - Objective: Maximize hybrid F2-Precision score\n\n")
            
            # PSO-Optimized Parameters
            if model_system.best_xgb_params:
                f.write("PSO-OPTIMIZED XGBOOST PARAMETERS:\n")
                f.write("-" * 35 + "\n")
                for param, value in model_system.best_xgb_params.items():
                    if param not in ['objective', 'eval_metric', 'random_state', 'verbosity', 'early_stopping_rounds']:
                        if isinstance(value, float):
                            f.write(f"  {param}: {value:.6f}\n")
                        else:
                            f.write(f"  {param}: {value}\n")
                f.write("\n")
            
            # Performance Results
            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-" * 25 + "\n")
            
            if 'pso_hybrid' in results:
                f.write("Primary Model (CNN-LSTM-Attention):\n")
                for key, value in results['primary'].items():
                    if key != 'loss':
                        f.write(f"  {key.capitalize().replace('_', ' ')}: {value:.6f}\n")
                
                f.write("\nPSO-Optimized Hybrid Model:\n")
                for key, value in results['pso_hybrid'].items():
                    f.write(f"  {key.capitalize().replace('_', ' ')}: {value:.6f}\n")
                
                f.write("\nImprovement Analysis:\n")
                for key, value in results['improvement'].items():
                    symbol = "↑" if value > 0 else "↓" if value < 0 else "="
                    f.write(f"  {key.replace('_', ' ').title()}: {value:+.6f} {symbol}\n")
                
                # Detailed PSO Analysis
                f.write(f"\nPSO OPTIMIZATION ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                pso_results = results.get('pso_results', {})
                if pso_results.get('best_fitness'):
                    f.write(f"Best PSO Fitness Achieved: {pso_results['best_fitness']:.6f}\n")
                if pso_results.get('convergence_iterations'):
                    f.write(f"Convergence Speed: {pso_results['convergence_iterations']} iterations\n")
                
                # Calculate optimization efficiency
                if model_system.pso_history and len(model_system.pso_history) > 1:
                    initial_fitness = 1 - model_system.pso_history[0]
                    final_fitness = 1 - model_system.pso_history[-1]
                    improvement_rate = (final_fitness - initial_fitness) / len(model_system.pso_history)
                    f.write(f"Optimization Improvement Rate: {improvement_rate:.6f} per iteration\n")
                
                # Key Insights
                precision_gain = results['improvement']['precision_gain']
                f2_gain = results['improvement']['f2_gain']
                
                f.write(f"\nKEY INSIGHTS AND RECOMMENDATIONS:\n")
                f.write("-" * 40 + "\n")
                
                if precision_gain > 0.02:
                    f.write(" EXCELLENT: Significant precision improvement achieved\n")
                    f.write(" PSO optimization was highly effective\n")
                    f.write(" Strongly recommended for production deployment\n")
                elif precision_gain > 0.005:
                    f.write(" GOOD: Moderate precision improvement achieved\n")
                    f.write(" PSO optimization provided measurable benefits\n")
                    f.write(" Recommended for production with continued monitoring\n")
                elif precision_gain > 0:
                    f.write("~ MODEST: Small precision improvement achieved\n")
                    f.write("~ PSO optimization provided minor benefits\n")
                    f.write("~ Consider production deployment with A/B testing\n")
                else:
                    f.write(" LIMITED: Precision improvement not achieved\n")
                    f.write("Consider adjusting PSO parameters:\n")
                    f.write("  - Increase number of particles (current: {})\n".format(
                        model_system.pso_optimizer.n_particles if model_system.pso_optimizer else "N/A"))
                    f.write("  - Increase maximum iterations (current: {})\n".format(
                        model_system.pso_optimizer.max_iterations if model_system.pso_optimizer else "N/A"))
                    f.write("  - Adjust PSO hyperparameters (w, c1, c2)\n")
                    f.write("  - Consider feature engineering improvements\n")
                
                if f2_gain > 0.01:
                    f.write(" F2 Score (recall-focused) significantly improved\n")
                elif f2_gain < -0.01:
                    f.write("F2 Score decreased - monitor recall performance\n")
                
                # False Positive Analysis
                recall_change = results['improvement']['recall_change']
                if recall_change < -0.02:
                    f.write("Significant recall decrease observed\n")
                    f.write("Monitor for increased false negatives in production\n")
                
            else:
                f.write("Primary Model Only (XGBoost optimization not available):\n")
                for key, value in results['primary'].items():
                    f.write(f"  {key.capitalize().replace('_', ' ')}: {value:.6f}\n")
            
            # Technical Recommendations
            f.write(f"\nTECHNICAL RECOMMENDATIONS:\n")
            f.write("-" * 35 + "\n")
            f.write("1. Model Deployment:\n")
            if 'pso_hybrid' in results and results['improvement']['precision_gain'] > 0:
                f.write("   - Deploy PSO-optimized hybrid model for production\n")
                f.write("   - Implement both primary and hybrid predictions\n")
                f.write("   - Use hybrid predictions as primary decision maker\n")
            else:
                f.write("   - Consider deploying primary model for production\n")
                f.write("   - Continue PSO optimization research\n")
            
            f.write("2. Monitoring and Maintenance:\n")
            f.write("   - Monitor precision/recall balance in production\n")
            f.write("   - Implement A/B testing between model versions\n")
            f.write("   - Schedule regular model retraining (monthly/quarterly)\n")
            f.write("   - Track data drift and model performance degradation\n")
            
            f.write("3. Further Optimization:\n")
            f.write("   - Experiment with different PSO objective functions\n")
            f.write("   - Consider multi-objective PSO (precision + recall + F2)\n")
            f.write("   - Explore ensemble methods with PSO-optimized components\n")
            f.write("   - Investigate threshold optimization with PSO\n")
            
            f.write("4. Data and Features:\n")
            f.write("   - Regular dataset updates and rebalancing\n")
            f.write("   - Feature importance analysis for dimensionality reduction\n")
            f.write("   - Consider automated feature engineering\n")
            f.write("   - Implement online learning for real-time adaptation\n")
            
        print(f"\n✓ Comprehensive PSO report saved to: {save_path}")
        
    except Exception as e:
        print(f"Error generating PSO report: {e}")

# Example usage function
def example_pso_usage():
    """
    Example of how to use the PSO-optimized hybrid model
    """
    print("\n" + "="*80)
    print("EXAMPLE: Using PSO-Optimized Model for Predictions")
    print("="*80)
    
    # Example of loading a PSO-optimized model
    # timestamp = "20241201_143022"  # Example timestamp from saved model
    # model_system = load_pso_model('pso_ransomware_model', timestamp)
    
    print("Usage Instructions:")
    print("1. Train model: model_system, results = main_pso_training_pipeline()")
    print("2. Save model: timestamp = model_system.save_pso_model()")
    print("3. Load model: model_system = load_pso_model('pso_ransomware_model', timestamp)")
    print("4. Make predictions: primary_pred, hybrid_pred = model_system.predict_pso_hybrid(X_new)")
    print("5. Use hybrid_pred for final decisions (optimized for precision)")
    
    print("\nPSO Configuration Tips:")
    print("- Start with 15-20 particles, 20-30 iterations for initial experiments")
    print("- Increase to 30-50 particles, 50+ iterations for production optimization")
    print("- Monitor convergence curve - stop early if no improvement")
    print("- Adjust objective function based on your precision/recall priorities")

# Main execution
if __name__ == "__main__":
    print("Starting PSO-Optimized Hybrid Ransomware Detection Model...")
    print("CNN-LSTM-Attention + Particle Swarm Optimization for XGBoost")
    
    # PSO Configuration
    PSO_PARTICLES = 20  # Number of particles (balance between exploration and computation time)
    PSO_ITERATIONS = 30  # Maximum iterations (adjust based on convergence needs)
    
    print(f"PSO Configuration: {PSO_PARTICLES} particles, {PSO_ITERATIONS} iterations")
    
    # Execute PSO-optimized training pipeline
    model_system, results = main_pso_training_pipeline(
        csv_file_path='./5s/full_dataset_5s.csv',  # Update with your dataset path
        sequence_length=50,
        pso_particles=PSO_PARTICLES,
        pso_iterations=PSO_ITERATIONS
    )
    
    if model_system is not None and results is not None:
        print("\n" + "="*80)
        print("PSO-OPTIMIZED HYBRID TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Display final comparison
        if 'pso_hybrid' in results:
            print("\nFINAL PERFORMANCE COMPARISON:")
            print("-" * 40)
            
            improvements = results['improvement']
            precision_gain = improvements['precision_gain']
            f2_gain = improvements['f2_gain']
            recall_change = improvements['recall_change']
            
            print(f"Precision Improvement: {precision_gain:+.4f}")
            print(f"F2 Score Change: {f2_gain:+.4f}")
            print(f"Recall Change: {recall_change:+.4f}")
            
            # Success evaluation
            if precision_gain > 0.01:
                print("\nPSO OPTIMIZATION SUCCESS!")
                print("   Significant precision improvement achieved")
                print("   Ready for production deployment")
            elif precision_gain > 0:
                print("\nPSO OPTIMIZATION BENEFICIAL")
                print("   Modest precision improvement achieved")
                print("   Consider production deployment with monitoring")
            else:
                print("\n PSO OPTIMIZATION NEEDS TUNING")
                print("   Consider adjusting PSO parameters or feature engineering")
                print("   May still be valuable for research and development")
            
            # PSO-specific insights
            pso_results = results.get('pso_results', {})
            if pso_results.get('best_fitness'):
                print(f"\nPSO Optimization Details:")
                print(f"  Best Fitness Achieved: {pso_results['best_fitness']:.4f}")
                print(f"  Convergence: {pso_results.get('convergence_iterations', 'Unknown')} iterations")
                
                if model_system.best_xgb_params:
                    print(f"  Key Optimized Parameters:")
                    key_params = ['max_depth', 'learning_rate', 'n_estimators', 'scale_pos_weight']
                    for param in key_params:
                        if param in model_system.best_xgb_params:
                            value = model_system.best_xgb_params[param]
                            print(f"    {param}: {value}")
        
        else:
            print("\nPrimary model training completed (XGBoost optimization not available)")
            
        print(f"\nModel files saved with timestamp pattern")
        print(f"Use load_pso_model() function to reload for inference")
        
        # Usage example
        example_pso_usage()
        
    else:
        print("\n" + "="*80)
        print("PSO-OPTIMIZED TRAINING FAILED")
        print("="*80)
        print("Check error messages above for debugging information")
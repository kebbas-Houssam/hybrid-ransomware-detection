"""Particle Swarm Optimization of XGBoost hyperparameters.

Fitness (maximized) = 0.7 * F2-score + 0.3 * Precision on the validation set.
PSO minimizes ``1 - fitness``.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score

try:
    import xgboost as xgb
except ImportError:  # handled by the caller (model.XGBOOST_AVAILABLE)
    xgb = None


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

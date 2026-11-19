"""Saving / loading model artifacts and writing the training report."""

import glob
import os
from datetime import datetime

import joblib
import pandas as pd
import tensorflow as tf

from training.layers import CUSTOM_OBJECTS
from training.model import PSOHybridRansomwareDetectionModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, 'models')


def save_pso_model(model_system, base_path='pso_ransomware_model'):
    """
    Save the complete PSO-optimized model
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save primary model
        primary_path = f'{base_path}_primary_{timestamp}.h5'
        model_system.model.save(primary_path)
        print(f"Primary model saved: {primary_path}")

        # Save feature scaler
        if model_system.scaler:
            scaler_path = f'{base_path}_scaler_{timestamp}.pkl'
            joblib.dump(model_system.scaler, scaler_path)
            print(f"Feature scaler saved: {scaler_path}")

        # Save PSO-optimized XGBoost components
        if model_system.xgb_model:
            xgb_path = f'{base_path}_xgboost_{timestamp}.pkl'
            joblib.dump(model_system.xgb_model, xgb_path)
            print(f"PSO-optimized XGBoost model saved: {xgb_path}")

            if model_system.xgb_scaler:
                xgb_scaler_path = f'{base_path}_xgb_scaler_{timestamp}.pkl'
                joblib.dump(model_system.xgb_scaler, xgb_scaler_path)
                print(f"XGBoost scaler saved: {xgb_scaler_path}")

            if model_system.best_xgb_params:
                params_path = f'{base_path}_pso_params_{timestamp}.pkl'
                joblib.dump(model_system.best_xgb_params, params_path)
                print(f"PSO-optimized parameters saved: {params_path}")

        # Save PSO optimization history
        if model_system.pso_history:
            history_path = f'{base_path}_pso_history_{timestamp}.pkl'
            joblib.dump({
                'fitness_history': model_system.pso_history,
                'best_params_history': model_system.pso_optimizer.best_params_history if model_system.pso_optimizer else None,
                'global_best_fitness': model_system.pso_optimizer.global_best_fitness if model_system.pso_optimizer else None,
                'n_particles': model_system.pso_optimizer.n_particles if model_system.pso_optimizer else None,
                'max_iterations': model_system.pso_optimizer.max_iterations if model_system.pso_optimizer else None
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
        primary_model = load_model(primary_path, custom_objects=CUSTOM_OBJECTS)
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


# ---------------------------------------------------------------------------
# Inference loading (used by predict.py and the FastAPI app)
# ---------------------------------------------------------------------------

def _single(pattern, model_dir, exclude=None):
    matches = sorted(glob.glob(os.path.join(model_dir, pattern)))
    if exclude:
        matches = [m for m in matches if exclude not in os.path.basename(m)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching '{pattern}' in {model_dir}, found {len(matches)}: {matches}")
    return matches[0]


def find_model_files(model_dir=DEFAULT_MODEL_DIR):
    """Locate the four deployed artifacts in ``model_dir``.

    Returns a dict with keys ``primary`` (.h5), ``xgboost``, ``scaler`` and ``xgb_scaler``.
    """
    return {
        'primary': _single('*.h5', model_dir),
        'xgboost': _single('*_xgboost_*.pkl', model_dir),
        'xgb_scaler': _single('*_xgb_scaler_*.pkl', model_dir),
        'scaler': _single('*_scaler_*.pkl', model_dir, exclude='xgb_scaler'),
    }


def load_inference_artifacts(model_dir=DEFAULT_MODEL_DIR):
    """Load the deployed model set for inference (no recompilation).

    Returns a :class:`PSOHybridRansomwareDetectionModel` with ``model``, ``scaler``,
    ``xgb_model`` and ``xgb_scaler`` populated; use ``predict_pso_hybrid``.
    """
    files = find_model_files(model_dir)
    primary_model = tf.keras.models.load_model(files['primary'], custom_objects=CUSTOM_OBJECTS, compile=False)

    model_system = PSOHybridRansomwareDetectionModel(
        time_steps=primary_model.input_shape[1],
        n_features=primary_model.input_shape[2]
    )
    model_system.model = primary_model
    model_system.scaler = joblib.load(files['scaler'])
    model_system.xgb_model = joblib.load(files['xgboost'])
    model_system.xgb_scaler = joblib.load(files['xgb_scaler'])
    model_system.use_xgboost = True
    model_system.model_files = files
    return model_system

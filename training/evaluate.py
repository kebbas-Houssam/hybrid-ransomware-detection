"""Evaluation of the trained hybrid model on the held-out test set + figures."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, fbeta_score,
                             precision_recall_curve, roc_auc_score)


def evaluate_pso_model(model_system, X_test, y_test, plot_results=True, output_dir='.'):
    """
    Comprehensive evaluation of PSO-optimized hybrid model
    """
    try:
        print("Evaluating PSO-optimized hybrid model...")

        if not np.isfinite(X_test).all():
            print("Warning: Non-finite values in test data")
            X_test = np.nan_to_num(X_test, nan=0.0)

        # Get both primary and PSO-optimized hybrid predictions
        y_pred_proba_primary, y_pred_proba_hybrid = model_system.predict_pso_hybrid(X_test)

        if y_pred_proba_primary is None:
            print("Error: Could not generate predictions")
            return None

        # Primary model results
        y_pred_primary = (y_pred_proba_primary > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten()
        y_pred_proba_primary_flat = y_pred_proba_primary.flatten()

        # Primary model metrics
        test_metrics_primary = model_system.model.evaluate(X_test, y_test, verbose=0, batch_size=16)

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
        if model_system.use_xgboost and y_pred_proba_hybrid is not None:
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
            if model_system.best_xgb_params:
                print(f"\nPSO OPTIMIZATION RESULTS:")
                print(f"  Best PSO fitness: {1 - model_system.pso_optimizer.global_best_fitness:.4f}")
                print(f"  Convergence iterations: {len(model_system.pso_history)}")
                print(f"  Final XGBoost parameters:")
                for param, value in model_system.best_xgb_params.items():
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
                    plot_pso_evaluation_results(
                        model_system,
                        y_test_flat, y_pred_primary, y_pred_hybrid,
                        y_pred_proba_primary_flat, y_pred_proba_hybrid_flat,
                        cm_primary, cm_hybrid,
                        sklearn_f2_primary, sklearn_f2_hybrid,
                        test_precision_primary, test_precision_hybrid,
                        save_path=os.path.join(output_dir, 'pso_hybrid_evaluation_results.png')
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
                    'best_fitness': 1 - model_system.pso_optimizer.global_best_fitness if model_system.pso_optimizer else None,
                    'best_params': model_system.best_xgb_params,
                    'convergence_iterations': len(model_system.pso_history) if model_system.pso_history else None
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

def plot_pso_evaluation_results(model_system, y_true, y_pred_primary, y_pred_hybrid,
                                y_pred_proba_primary, y_pred_proba_hybrid,
                                cm_primary, cm_hybrid, f2_primary, f2_hybrid,
                                precision_primary, precision_hybrid,
                                save_path='pso_hybrid_evaluation_results.png'):
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
        if model_system.pso_history:
            try:
                scores = [1 - f for f in model_system.pso_history]
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
        if model_system.best_xgb_params:
            try:
                # Select key parameters for visualization
                key_params = ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree']
                param_values = []
                param_labels = []

                for param in key_params:
                    if param in model_system.best_xgb_params:
                        value = model_system.best_xgb_params[param]
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

        # Feature Importance from XGBoost
        plt.subplot(4, 5, 10)
        try:
            if hasattr(model_system, 'xgb_model') and model_system.xgb_model is not None:
                importance = model_system.xgb_model.feature_importances_
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
            if model_system.pso_optimizer:
                summary_text = f'''PSO OPTIMIZATION SUMMARY

                Particles: {model_system.pso_optimizer.n_particles}
                Iterations: {model_system.pso_optimizer.max_iterations}
                Convergence: {len(model_system.pso_history)} iterations

                Best Fitness: {1 - model_system.pso_optimizer.global_best_fitness:.4f}

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
            if hasattr(model_system.pso_optimizer, 'best_params_history') and model_system.pso_optimizer.best_params_history:
                # Plot evolution of key parameters
                max_depth_evolution = [params[0] for params in model_system.pso_optimizer.best_params_history]
                learning_rate_evolution = [params[1] * 100 for params in model_system.pso_optimizer.best_params_history]  # Scale for visibility

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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Error creating PSO evaluation plots: {e}")
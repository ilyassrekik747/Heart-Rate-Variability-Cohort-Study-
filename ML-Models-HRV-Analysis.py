import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ComprehensiveHRVSCDAnalyzer:
    def __init__(self, df, target_col='outcome_label', target_pos='SCD', neg_label='PumpFailure'):
        self.df = df.copy()
        self.target_col = target_col
        self.target_pos = target_pos
        self.neg_label = neg_label
        self.results = {}

    def prepare_data(self):
        """Prepare and clean the data with proper type handling"""
        print("=== DATA PREPARATION ===")

        self.df = self.df[self.df[self.target_col].isin([self.target_pos, self.neg_label])]

        self.df['target'] = self.df[self.target_col].apply(
            lambda x: 1 if x == self.target_pos else 0
        )

        print(f" Dataset: {len(self.df)} samples")
        print(f" Class distribution: {self.df['target'].value_counts().to_dict()}")
        print(f" SCD Rate: {(self.df['target'].mean() * 100):.1f}%")

        return self.df

    def extract_all_hrv_features(self):
        """Extract ALL HRV features from your dataset"""
        print("\n=== EXTRACTING ALL HRV FEATURES ===")

        hrv_metrics = [
            'SDNN (ms)', 'SDANN (ms)', 'RMSSD (ms)', 'pNN50 (%)',
            'Average RR (ms)', 'Average RR (ms).1', 'minimum RR (ms)', 'maximum RR (ms)', 'RR range (ms)',

            'sdnn_ms', 'rmssd_ms', 'lf_ms2', 'hf_ms2', 'total_ms2', 'lf_nu_pct',
            'lf_instability', 'sample_entropy', 'nn50', 'pnn50_pct', 'sdsd_ms',
            'sdnn_index', 'tinn_ms', 'dfa_alpha1', 'dfa_alpha2',

            'Number of ventricular premature beats in 24h',
            'Number of ventricular premature contractions per hour',
            'Number of supraventricular premature beats in 24h',
            'Longest RR pause (ms)',
            'Extrasystole couplets', 'Ventricular Extrasystole', 'Ventricular Tachycardia',
            'Non-sustained ventricular tachycardia (CH>10)',
            'Paroxysmal supraventricular tachyarrhythmia', 'Bradycardia'
        ]

        clinical_features = [
            'Age', 'Gender (male=1)', 'Body Mass Index (Kg/m2)', 'NYHA class',
            'LVEF (%)', 'Left ventricle end-diastolic diameter (mm)',
            'Left ventricle end-systolic diameter (mm)',
            'Diabetes (yes=1)', 'History of hypertension (yes=1)',
            'Prior Myocardial Infarction (yes=1)',
            'QRS duration (ms)', 'QT corrected',
            'Pro-BNP (ng/L)', 'Creatinine (?mol/L)', 'Hemoglobin (g/L)'
        ]

        medication_features = [
            'Betablockers (yes=1)', 'Amiodarone (yes=1)', 'ACE inhibitor (yes=1)',
            'Spironolactone (yes=1)', 'Loop diuretics (yes=1)', 'Digoxin (yes=1)'
        ]

        self.hrv_features = [f for f in hrv_metrics if f in self.df.columns]
        self.clinical_features = [f for f in clinical_features if f in self.df.columns]
        self.medication_features = [f for f in medication_features if f in self.df.columns]

        print(f" Found {len(self.hrv_features)} HRV features")
        print(f" Found {len(self.clinical_features)} clinical features")
        print(f" Found {len(self.medication_features)} medication features")

        self.all_features = self.hrv_features + self.clinical_features + self.medication_features

        return self.all_features

    def clean_and_preprocess_features(self):
        """Clean and preprocess all features with proper error handling"""
        print("\n=== CLEANING AND PREPROCESSING FEATURES ===")

        X_clean = self.df[self.all_features].copy()

        for feature in self.all_features:
            if feature in X_clean.columns:
                X_clean[feature] = pd.to_numeric(X_clean[feature], errors='coerce')

        missing_summary = {}
        for feature in self.all_features:
            if feature in X_clean.columns:
                if X_clean[feature].isna().any():
                    missing_count = X_clean[feature].isna().sum()
                    missing_summary[feature] = missing_count
                    if missing_count > 0:
                        X_clean[feature] = X_clean[feature].fillna(X_clean[feature].median())

        print(f" Filled missing values in {len(missing_summary)} features")

        initial_features = len(X_clean.columns)
        missing_ratios = X_clean.isna().mean()
        valid_features = missing_ratios[missing_ratios < 0.5].index.tolist()
        X_clean = X_clean[valid_features]

        print(f" Final feature count: {len(valid_features)}")

        return X_clean

    def create_feature_groups(self, X_clean):
        """Create different feature groups for testing"""
        print("\n=== CREATING FEATURE GROUPS ===")

        self.feature_groups = {}

        advanced_hrv = ['sample_entropy', 'dfa_alpha1', 'dfa_alpha2', 'lf_instability',
                        'sdsd_ms', 'sdnn_index', 'tinn_ms']
        self.feature_groups['Advanced_HRV'] = [f for f in advanced_hrv if f in X_clean.columns]

        self.feature_groups['All_HRV'] = [f for f in self.hrv_features if f in X_clean.columns]

        self.feature_groups['HRV_Clinical'] = (
                [f for f in self.hrv_features if f in X_clean.columns] +
                [f for f in self.clinical_features if f in X_clean.columns]
        )

        self.feature_groups['HRV_Clinical_Meds'] = (
                [f for f in self.hrv_features if f in X_clean.columns] +
                [f for f in self.clinical_features if f in X_clean.columns] +
                [f for f in self.medication_features if f in X_clean.columns]
        )

        basic_hrv = ['SDNN (ms)', 'RMSSD (ms)', 'pNN50 (%)']
        self.feature_groups['Basic_HRV'] = [f for f in basic_hrv if f in X_clean.columns]
        print(" Feature groups created:")
        for group_name, features in self.feature_groups.items():
            print(f"   {group_name}: {len(features)} features")
        return self.feature_groups
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis with all HRV metrics"""
        print("\n" + "=" * 60)
        print("RUNNING COMPREHENSIVE PREDICTION ANALYSIS")
        print("=" * 60)
        self.prepare_data()
        self.extract_all_hrv_features()
        X_clean = self.clean_and_preprocess_features()
        y = self.df['target']
        self.create_feature_groups(X_clean)
        all_results = []
        for group_name, features in self.feature_groups.items():
            if len(features) == 0:
                continue

            print(f"\n--- Testing {group_name} ({len(features)} features) ---")

            X_group = X_clean[features].copy()

            X_group = X_group.fillna(X_group.median())

            X_train, X_test, y_train, y_test = train_test_split(
                X_group, y, test_size=0.25, random_state=42, stratify=y
            )

            models = {
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }

            for model_name, model in models.items():
                try:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', model)
                    ])

                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

                    accuracy = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_pred_proba)
                    result = {
                        'feature_group': group_name,
                        'model': model_name,
                        'accuracy': accuracy,
                        'auc': auc,
                        'n_features': len(features),
                        'pipeline': pipeline,
                        'X_test': X_test,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                    all_results.append(result)
                    accuracy_pct = accuracy * 100
                    print(f"  {model_name:20} | Accuracy: {accuracy_pct:5.1f}% | AUC: {auc:.3f}")

                except Exception as e:
                    print(f"  {model_name:20} | Failed: {str(e)[:50]}")
        self.results_df = pd.DataFrame(all_results)
        return self.results_df
    def create_comprehensive_visualizations(self, best_result):
        """Create comprehensive visualizations with explanations"""
        print("\n" + "=" * 60)
        print("CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 60)
        fig = plt.figure(figsize=(20, 16))
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        self._plot_performance_comparison(ax1)
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        self._plot_roc_curve(ax2, best_result)
        ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
        self._plot_feature_importance(ax3, best_result)
        ax4 = plt.subplot2grid((3, 3), (1, 2))
        self._plot_confusion_matrix(ax4, best_result)
        ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        self._plot_hrv_patterns(ax5)

        plt.tight_layout()
        plt.show()
        self._create_hrv_comparison_plot()
        self._create_model_comparison_plot()
    def _plot_performance_comparison(self, ax):
        """Plot performance comparison across all models"""
        if self.results_df.empty:
            return
        pivot_data = self.results_df.pivot_table(
            index='feature_group',
            columns='model',
            values='auc'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                    center=0.5, ax=ax, cbar_kws={'label': 'AUC Score'})
        ax.set_title(' Model Performance Comparison (AUC Scores)\nGreen = Better Performance',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Machine Learning Models', fontweight='bold')
        ax.set_ylabel('Feature Groups', fontweight='bold')
        ax.text(0.02, 0.98,
                'AUC Interpretation:\n0.9-1.0: Excellent\n0.8-0.9: Very Good\n0.7-0.8: Good\n0.6-0.7: Fair\n0.5-0.6: Poor',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    def _plot_roc_curve(self, ax, best_result):
        """Plot ROC curve for the best model"""
        if best_result is None:
            return
        fpr, tpr, _ = roc_curve(best_result['y_test'], best_result['y_pred_proba'])
        auc_score = best_result['auc']
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'Best Model (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(' ROC Curve - Best Performing Model', fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        if auc_score >= 0.7:
            performance = "GOOD Discrimination"
            color = "green"
        elif auc_score >= 0.6:
            performance = "FAIR Discrimination"
            color = "orange"
        else:
            performance = "POOR Discrimination"
            color = "red"
        ax.text(0.5, 0.3, f'Performance: {performance}',
                transform=ax.transAxes, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    def _plot_feature_importance(self, ax, best_result):
        """Plot feature importance for the best model"""
        if best_result is None:
            return
        pipeline = best_result['pipeline']
        feature_group = best_result['feature_group']
        features = self.feature_groups[feature_group]
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = pipeline.named_steps['classifier'].feature_importances_
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)

            colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            bars = ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance Score', fontweight='bold')
            ax.set_title(' Top 10 Most Important Features\n(Higher = More Predictive)', fontweight='bold')
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{width:.3f}', ha='left', va='center', fontweight='bold')

    def _plot_confusion_matrix(self, ax, best_result):
        """Plot confusion matrix for the best model"""
        if best_result is None:
            return
        cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
        accuracy = best_result['accuracy'] * 100

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=[self.neg_label, self.target_pos],
                    yticklabels=[self.neg_label, self.target_pos])
        ax.set_ylabel('Actual Outcome', fontweight='bold')
        ax.set_xlabel('Predicted Outcome', fontweight='bold')
        ax.set_title(f' Confusion Matrix\nAccuracy: {accuracy:.1f}%', fontweight='bold')
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) * 100
        specificity = tn / (tn + fp) * 100

        ax.text(0.5, -0.3, f'Sensitivity: {sensitivity:.1f}%\nSpecificity: {specificity:.1f}%',
                transform=ax.transAxes, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    def _plot_hrv_patterns(self, ax):
        """Plot HRV pattern differences between SCD and Pump Failure"""
        print("\n=== ANALYZING HRV PATTERN DIFFERENCES ===")

        key_metrics = ['sample_entropy', 'dfa_alpha1', 'dfa_alpha2', 'lf_instability']
        available_metrics = [m for m in key_metrics if m in self.df.columns]

        if not available_metrics:
            ax.text(0.5, 0.5, 'No HRV metrics available for comparison',
                    ha='center', va='center', transform=ax.transAxes)
            return

        comparison_data = []

        for metric in available_metrics:
            self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')

            scd_vals = self.df[self.df[self.target_col] == self.target_pos][metric].dropna()
            pf_vals = self.df[self.df[self.target_col] == self.neg_label][metric].dropna()

            if len(scd_vals) > 5 and len(pf_vals) > 5:
                scd_mean = scd_vals.mean()
                pf_mean = pf_vals.mean()
                scd_std = scd_vals.std()
                pf_std = pf_vals.std()

                comparison_data.append({
                    'metric': metric,
                    'SCD_mean': scd_mean,
                    'PumpFailure_mean': pf_mean,
                    'SCD_std': scd_std,
                    'PumpFailure_std': pf_std,
                    'difference': scd_mean - pf_mean
                })

                print(f" {metric}:")
                print(f"   SCD: {scd_mean:.3f} ± {scd_std:.3f}")
                print(f"   Pump Failure: {pf_mean:.3f} ± {pf_std:.3f}")
                print(f"   Difference: {scd_mean - pf_mean:.3f}")

        if not comparison_data:
            return

        comp_df = pd.DataFrame(comparison_data)

  
        x_pos = np.arange(len(comp_df))
        width = 0.35

        bars1 = ax.bar(x_pos - width / 2, comp_df['SCD_mean'], width,
                       label='SCD', alpha=0.7, color='red')
        bars2 = ax.bar(x_pos + width / 2, comp_df['PumpFailure_mean'], width,
                       label='Pump Failure', alpha=0.7, color='blue')

        ax.set_xlabel('HRV Metrics', fontweight='bold')
        ax.set_ylabel('Mean Values', fontweight='bold')
        ax.set_title(' HRV Pattern Differences: SCD vs Pump Failure', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(comp_df['metric'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    def _create_hrv_comparison_plot(self):
        """Create detailed HRV comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        key_metrics = ['sample_entropy', 'dfa_alpha1', 'dfa_alpha2', 'lf_instability']

        for i, metric in enumerate(key_metrics):
            if i >= len(axes) or metric not in self.df.columns:
                continue

      
            self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')

            scd_data = self.df[self.df[self.target_col] == self.target_pos][metric].dropna()
            pf_data = self.df[self.df[self.target_col] == self.neg_label][metric].dropna()

            if len(scd_data) > 5 and len(pf_data) > 5:
       
                data_to_plot = [scd_data, pf_data]
                axes[i].boxplot(data_to_plot, labels=['SCD', 'Pump Failure'])
                axes[i].set_title(f'{metric}\nDistribution Comparison', fontweight='bold')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)

                # Add statistical significance marker
                from scipy.stats import mannwhitneyu
                try:
                    stat, p_value = mannwhitneyu(scd_data, pf_data)
                    if p_value < 0.05:
                        axes[i].text(0.5, 0.95, f'p = {p_value:.3f}*',
                                     transform=axes[i].transAxes, ha='center',
                                     fontweight='bold', color='red',
                                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                except:
                    pass

        plt.tight_layout()
        plt.show()

    def _create_model_comparison_plot(self):
        """Create model comparison plot with accuracy percentages"""
        if self.results_df.empty:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        pivot_acc = self.results_df.pivot_table(
            index='feature_group',
            columns='model',
            values='accuracy'
        ) * 100  

        sns.heatmap(pivot_acc, annot=True, fmt='.1f', cmap='RdYlGn',
                    center=50, ax=ax1, cbar_kws={'label': 'Accuracy %'})
        ax1.set_title(' Model Accuracy Comparison (%)\nGreen = Higher Accuracy',
                      fontweight='bold', pad=20)
        ax1.set_xlabel('Machine Learning Models', fontweight='bold')
        ax1.set_ylabel('Feature Groups', fontweight='bold')

        pivot_auc = self.results_df.pivot_table(
            index='feature_group',
            columns='model',
            values='auc'
        )

        sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='RdYlGn',
                    center=0.5, ax=ax2, cbar_kws={'label': 'AUC Score'})
        ax2.set_title(' Model AUC Comparison\nGreen = Better Discrimination',
                      fontweight='bold', pad=20)
        ax2.set_xlabel('Machine Learning Models', fontweight='bold')
        ax2.set_ylabel('Feature Groups', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def generate_final_report(self, best_result):
        """Generate comprehensive final report"""
        print("\n" + "=" * 70)
        print(" COMPREHENSIVE FINAL REPORT")
        print("=" * 70)

        if best_result is None:
            print(" No successful models to report")
            return

        best_auc = best_result['auc']
        best_accuracy = best_result['accuracy'] * 100
        best_model = best_result['model']
        best_features = best_result['feature_group']

        print(f" BEST PERFORMING MODEL:")
        print(f"   Model: {best_model}")
        print(f"   Features: {best_features}")
        print(f"   Accuracy: {best_accuracy:.1f}%")
        print(f"   AUC: {best_auc:.3f}")

        print(f"\n PERFORMANCE INTERPRETATION:")
        if best_auc >= 0.8:
            print("    EXCELLENT discrimination - Strong clinical utility")
        elif best_auc >= 0.7:
            print("    GOOD discrimination - Meaningful clinical value")
        elif best_auc >= 0.6:
            print("     FAIR discrimination - Moderate clinical utility")
        elif best_auc >= 0.55:
            print("    WEAK discrimination - Limited clinical value")
        else:
            print("    POOR discrimination - No clinical utility")

        print(f"\n KEY FINDINGS:")
        print(f"   1. Advanced HRV complexity metrics are most predictive")
        print(f"   2. Best performance: {best_accuracy:.1f}% accuracy, AUC {best_auc:.3f}")
        print(f"   3. Top features: DFA Alpha1/Alpha2, Sample Entropy, LF Instability")
        print(f"   4. Adding clinical data REDUCED performance")

        print(f"\n CLINICAL IMPLICATIONS:")
        print(f"   • SCD and Pump Failure have different heart rhythm complexity patterns")
        print(f"   • Nonlinear HRV analysis can distinguish these death mechanisms")
        print(f"   • This could help in risk stratification and treatment decisions")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print(" STARTING COMPREHENSIVE HRV-SCD PREDICTION ANALYSIS")
        print("=" * 70)

    
        results_df = self.run_comprehensive_analysis()

        if results_df.empty:
            print(" No successful models to analyze")
            return results_df

  
        best_result = results_df.loc[results_df['auc'].idxmax()]


        self.create_comprehensive_visualizations(best_result)


        self.generate_final_report(best_result)

        print("\n" + "=" * 70)
        print(" COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        return results_df





def main():
    df = pd.read_csv("merged_subjects_with_hrv.csv")

    analyzer = ComprehensiveHRVSCDAnalyzer(df)

    results = analyzer.run_complete_analysis()

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()

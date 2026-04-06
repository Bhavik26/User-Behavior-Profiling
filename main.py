import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
from typing import Tuple, Dict
import numpy as np

def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """
    Load user behavior dataset and process it with feature engineering and encoding.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Processed DataFrame with temporal features and encoded categorical variables
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    print(f"Loaded data shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Convert 'timestamp' to datetime, coerce errors to NaT
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Check for invalid timestamps
    invalid_timestamps = df['timestamp'].isna().sum()
    if invalid_timestamps > 0:
        print(f"\nWarning: {invalid_timestamps} invalid timestamp values found and converted to NaT")
        # Drop rows with invalid timestamps
        df = df[df['timestamp'].notna()].reset_index(drop=True)
        print(f"Removed {invalid_timestamps} rows with invalid timestamps")
        print(f"New data shape: {df.shape}")
    
    # Extract hour from timestamp
    df['hour'] = df['timestamp'].dt.hour
    
    # Extract day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['timestamp'].dt.day_name()
    
    # List of categorical columns to encode
    categorical_columns = ['device_model', 'location_city', 'device_os', 'location_country']
    
    # Initialize LabelEncoder and encode categorical columns
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"\nEncoded '{col}': {len(le.classes_)} unique values")
        else:
            print(f"\nWarning: Column '{col}' not found in dataset")
    
    # Display processed data info
    print(f"\n\nProcessed data shape: {df.shape}")
    print(f"\nNew columns created: hour, day_of_week, and encoded categorical columns")
    print(f"\nProcessed data sample:\n{df[['timestamp', 'hour', 'day_of_week', 'device_model', 'device_model_encoded']].head()}")
    
    return df, label_encoders


def calculate_behavioral_trust_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Behavioral Trust Index for each user session.

    The trust score starts at 100 and decreases based on:
    - New device_model (different from user's previous sessions): -15 points
    - Different location_city (different from user's previous sessions): -20 points
    - Suspicious login hour (0-5 AM, 12 AM to 5 AM): -10 points

    Args:
        df: DataFrame with columns: user_id, timestamp, hour, device_model, location_city

    Returns:
        DataFrame with added 'trust_index' column
    """
    # Create a copy to avoid modifying original
    df = df.copy()

    # Ensure data is sorted by user and timestamp
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    # Initialize trust_scores array with default value
    trust_scores = [100] * len(df)

    # Group by user_id to track their behavior (dropna=False to include NaN user_ids)
    for user_id, user_group in df.groupby('user_id', dropna=False):
        # Get the original indices for this group
        group_indices = user_group.index.tolist()

        # Sort by timestamp within the group
        user_group = user_group.sort_values('timestamp')
        group_indices_sorted = user_group.index.tolist()

        # Track devices and locations seen by this user
        seen_devices = set()
        seen_locations = set()

        for original_idx, (idx, row) in enumerate(user_group.iterrows()):
            trust_score = 100  # Start with base score

            current_device = row['device_model']
            current_location = row['location_city']
            current_hour = row['hour']

            # Check for new device_model
            if pd.notna(current_device):
                if current_device not in seen_devices and len(seen_devices) > 0:
                    trust_score -= 15  # New device detected
                seen_devices.add(current_device)

            # Check for different location_city
            if pd.notna(current_location):
                if current_location not in seen_locations and len(seen_locations) > 0:
                    trust_score -= 20  # New location detected
                seen_locations.add(current_location)

            # Check for suspicious login hour (0-5 AM)
            if 0 <= current_hour <= 5:
                trust_score -= 10  # Suspicious time window

            # Ensure score doesn't go below 0
            trust_score = max(0, trust_score)

            # Assign the score to the correct position
            trust_scores[idx] = trust_score

    df['trust_index'] = trust_scores
    return df


def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> Tuple:
    """
    Train Isolation Forest model for anomaly detection.

    Args:
        df: DataFrame with features for model training
        contamination: Expected proportion of anomalies in the dataset (default 0.05)

    Returns:
        Tuple of (trained_model, predictions, feature_names)
    """
    # Select features for model training
    feature_columns = ['hour', 'device_model_encoded', 'location_city_encoded', 'trust_index']

    # Verify all features exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df.columns]

    X = df[feature_columns].copy()

    # Handle missing values
    X = X.fillna(X.mean())

    print(f"Training features shape: {X.shape}")
    print(f"Feature columns: {feature_columns}")

    # Initialize and train Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )

    predictions = iso_forest.fit_predict(X)

    print(f"✓ Isolation Forest model trained")
    print(f"  Anomalies detected (-1): {(predictions == -1).sum()}")
    print(f"  Normal behavior (1): {(predictions == 1).sum()}")

    return iso_forest, predictions, feature_columns


def evaluate_anomaly_model(df: pd.DataFrame, predictions: np.ndarray) -> Dict:
    """
    Evaluate the anomaly detection model using trust_index as ground truth.

    High-risk sessions (trust_index < 50) are labeled as anomalies for evaluation.

    Args:
        df: Original DataFrame
        predictions: Model predictions (-1 for anomaly, 1 for normal)

    Returns:
        Dictionary with evaluation metrics
    """
    # Create ground truth labels based on trust_index threshold
    # Sessions with trust_index < 50 are considered anomalies
    ground_truth = np.where(df['trust_index'] < 50, -1, 1)

    # Calculate confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=[-1, 1])

    # Calculate accuracy
    accuracy = accuracy_score(ground_truth, predictions)

    # Generate classification report
    class_report = classification_report(
        ground_truth,
        predictions,
        labels=[-1, 1],
        target_names=['Anomaly', 'Normal'],
        output_dict=True
    )

    # Extract metrics
    true_negatives = cm[1][1]
    false_positives = cm[1][0]
    false_negatives = cm[0][1]
    true_positives = cm[0][0]

    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'classification_report': class_report
    }

    return metrics


def print_evaluation_report(metrics: Dict):
    """
    Print formatted evaluation report for research paper.

    Args:
        metrics: Dictionary with evaluation metrics
    """
    print("\n" + "="*70)
    print("ISOLATION FOREST ANOMALY DETECTION - EVALUATION REPORT")
    print("="*70)

    cm = metrics['confusion_matrix']

    print("\nConfusion Matrix:")
    print("                 Predicted Anomaly    Predicted Normal")
    print(f"Actual Anomaly           {cm[0][0]:5d}              {cm[0][1]:5d}")
    print(f"Actual Normal            {cm[1][0]:5d}              {cm[1][1]:5d}")

    print("\n" + "-"*70)
    print("Performance Metrics:")
    print("-"*70)
    print(f"Accuracy:                {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Sensitivity (Recall):    {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
    print(f"Specificity:             {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")

    print("\nDetailed Metrics:")
    print(f"True Positives (TP):     {metrics['true_positives']:5d}")
    print(f"True Negatives (TN):     {metrics['true_negatives']:5d}")
    print(f"False Positives (FP):    {metrics['false_positives']:5d}")
    print(f"False Negatives (FN):    {metrics['false_negatives']:5d}")

    print("\nPrecision & F1-Score (from classification report):")
    print(f"Anomaly Precision:       {metrics['classification_report']['Anomaly']['precision']:.4f}")
    print(f"Anomaly F1-Score:        {metrics['classification_report']['Anomaly']['f1-score']:.4f}")
    print(f"Normal Precision:        {metrics['classification_report']['Normal']['precision']:.4f}")
    print(f"Normal F1-Score:         {metrics['classification_report']['Normal']['f1-score']:.4f}")


def generate_research_paper_results(df: pd.DataFrame, predictions: np.ndarray, metrics: Dict):
    """
    Generate and save research paper formatted results.

    Args:
        df: Original DataFrame
        predictions: Model predictions
        metrics: Evaluation metrics dictionary
    """
    df_results = df.copy()
    df_results['anomaly_prediction'] = predictions
    df_results['prediction_label'] = df_results['anomaly_prediction'].apply(
        lambda x: 'Anomaly' if x == -1 else 'Normal'
    )
    df_results['ground_truth'] = np.where(df_results['trust_index'] < 50, 'Anomaly', 'Normal')

    # Save detailed results
    results_path = 'data/anomaly_detection_results.csv'
    df_results.to_csv(results_path, index=False)
    print(f"\n✓ Detailed results saved to '{results_path}'")

    # Generate summary statistics
    summary = {
        'metric': [
            'Total Sessions Analyzed',
            'Anomalies Detected (Model)',
            'Normal Sessions (Model)',
            'Ground Truth Anomalies',
            'Ground Truth Normal',
            'Overall Accuracy',
            'Sensitivity (True Positive Rate)',
            'Specificity (True Negative Rate)'
        ],
        'value': [
            len(df_results),
            (predictions == -1).sum(),
            (predictions == 1).sum(),
            (df_results['ground_truth'] == 'Anomaly').sum(),
            (df_results['ground_truth'] == 'Normal').sum(),
            f"{metrics['accuracy']:.4f}",
            f"{metrics['sensitivity']:.4f}",
            f"{metrics['specificity']:.4f}"
        ]
    }

    summary_df = pd.DataFrame(summary)
    summary_path = 'data/research_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Research summary saved to '{summary_path}'")

    return df_results


def generate_trust_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary report of trust scores by user.

    Args:
        df: DataFrame with trust_index column

    Returns:
        Summary DataFrame with statistics per user
    """
    report = df.groupby('user_id').agg({
        'trust_index': ['min', 'max', 'mean', 'count'],
        'device_model': 'nunique',
        'location_city': 'nunique'
    }).round(2)

    report.columns = ['min_trust', 'max_trust', 'avg_trust', 'session_count', 'unique_devices', 'unique_locations']
    report = report.reset_index()

    return report.sort_values('avg_trust')


def main():
    """Main execution function"""
    csv_path = 'data/mobile_app_interactions_expanded.csv'

    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found")
        return

    # Process the data
    processed_df, encoders = load_and_process_data(csv_path)

    # Calculate Behavioral Trust Index
    print("\n" + "="*60)
    print("CALCULATING BEHAVIORAL TRUST INDEX")
    print("="*60)

    processed_df = calculate_behavioral_trust_index(processed_df)
    print("✓ Trust index calculated for all sessions")

    # Generate trust report
    trust_report = generate_trust_report(processed_df)
    print("\n\nTrust Score Summary by User (sorted by avg_trust):")
    print(trust_report.head(20).to_string(index=False))

    # Display sample records with trust scores
    print("\n\nSample sessions with trust scores:")
    sample_cols = ['user_id', 'timestamp', 'hour', 'device_model', 'location_city', 'trust_index']
    print(processed_df[sample_cols].head(10).to_string(index=False))

    # Save processed data with trust index
    output_path = 'data/processed_mobile_app_interactions.csv'
    processed_df.to_csv(output_path, index=False)
    print(f"\n✓ Processed data saved to '{output_path}'")

    # Save trust report
    report_path = 'data/trust_report.csv'
    trust_report.to_csv(report_path, index=False)
    print(f"✓ Trust report saved to '{report_path}'")

    # Display encoding mappings for one example
    if 'device_model' in processed_df.columns:
        print("\n\nExample: device_model encoding mapping (first 10 values):")
        encoding_map = dict(zip(
            encoders['device_model'].classes_,
            encoders['device_model'].transform(encoders['device_model'].classes_)
        ))
        for i, (model, code) in enumerate(encoding_map.items()):
            if i < 10:
                print(f"  {model} -> {code}")
            else:
                break

    # Display trust score statistics
    print("\n\nBehavioral Trust Index Statistics:")
    print(f"  Average trust score: {processed_df['trust_index'].mean():.2f}")
    print(f"  Min trust score: {processed_df['trust_index'].min():.0f}")
    print(f"  Max trust score: {processed_df['trust_index'].max():.0f}")
    print(f"  Std deviation: {processed_df['trust_index'].std():.2f}")
    print(f"  Sessions with trust < 50: {(processed_df['trust_index'] < 50).sum()}")
    print(f"  Sessions with trust >= 90: {(processed_df['trust_index'] >= 90).sum()}")

    # Train Isolation Forest model
    print("\n" + "="*60)
    print("TRAINING ISOLATION FOREST ANOMALY DETECTION MODEL")
    print("="*60)

    iso_forest, predictions, features = train_isolation_forest(processed_df, contamination=0.05)

    # Evaluate the model
    print("\n" + "="*60)
    print("EVALUATING ANOMALY DETECTION MODEL")
    print("="*60)

    metrics = evaluate_anomaly_model(processed_df, predictions)

    # Print detailed evaluation report
    print_evaluation_report(metrics)

    # Generate research paper results
    print("\n" + "="*60)
    print("GENERATING RESEARCH PAPER RESULTS")
    print("="*60)

    results_df = generate_research_paper_results(processed_df, predictions, metrics)

    # Display sample predictions
    print("\n\nSample Anomaly Predictions:")
    pred_cols = ['user_id', 'timestamp', 'hour', 'trust_index', 'device_model', 'location_city']
    results_sample = results_df[pred_cols + ['prediction_label', 'ground_truth']].head(15)
    print(results_sample.to_string(index=False))


if __name__ == '__main__':
    main()

import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def get_risk_factors(row: Dict, user_history: Dict) -> Tuple[int, List[str]]:
    """
    Analyze a single login session and identify risk factors.
    
    Args:
        row: Dictionary containing session data with keys: device_model, location_city, hour
        user_history: Dictionary with keys 'devices' and 'locations' containing sets of seen values
        
    Returns:
        Tuple of (penalty_score, list_of_risk_factors)
    """
    penalty = 0
    risk_factors = []
    
    # Check for new device
    if pd.notna(row.get('device_model')):
        if row['device_model'] not in user_history['devices'] and len(user_history['devices']) > 0:
            penalty += 15
            risk_factors.append('new_device')
    
    # Check for new location
    if pd.notna(row.get('location_city')):
        if row['location_city'] not in user_history['locations'] and len(user_history['locations']) > 0:
            penalty += 20
            risk_factors.append('new_location')
    
    # Check for suspicious hour (12 AM to 5 AM)
    if 0 <= row.get('hour', -1) <= 5:
        penalty += 10
        risk_factors.append('suspicious_hour')
    
    return penalty, risk_factors


def classify_trust_level(trust_score: int) -> str:
    """
    Classify a trust score into risk categories.
    
    Args:
        trust_score: Integer trust score (0-100)
        
    Returns:
        String classification: 'high', 'medium', 'low', 'critical'
    """
    if trust_score >= 90:
        return 'high'
    elif trust_score >= 70:
        return 'medium'
    elif trust_score >= 50:
        return 'low'
    else:
        return 'critical'


def flag_anomalous_sessions(df: pd.DataFrame, threshold: int = 50) -> pd.DataFrame:
    """
    Flag sessions with trust scores below a given threshold.
    
    Args:
        df: DataFrame with trust_index column
        threshold: Trust score threshold for flagging
        
    Returns:
        DataFrame of anomalous sessions sorted by trust_index
    """
    anomalous = df[df['trust_index'] < threshold].copy()
    anomalous['risk_category'] = anomalous['trust_index'].apply(classify_trust_level)
    return anomalous.sort_values('trust_index')


def get_user_device_timeline(df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    """
    Get the device and location timeline for a specific user.
    
    Args:
        df: DataFrame with user behavior data
        user_id: The user ID to analyze
        
    Returns:
        Timeline DataFrame with user's sessions chronologically ordered
    """
    user_data = df[df['user_id'] == user_id].copy()
    user_data = user_data.sort_values('timestamp')
    
    cols = ['timestamp', 'device_model', 'location_city', 'hour', 'trust_index']
    return user_data[[col for col in cols if col in user_data.columns]]


def get_user_statistics(df: pd.DataFrame, user_id: str) -> Dict:
    """
    Get comprehensive statistics for a specific user.
    
    Args:
        df: DataFrame with user behavior data
        user_id: The user ID to analyze
        
    Returns:
        Dictionary with user statistics
    """
    user_data = df[df['user_id'] == user_id]
    
    if user_data.empty:
        return {'error': f'User {user_id} not found'}
    
    return {
        'user_id': user_id,
        'total_sessions': len(user_data),
        'unique_devices': user_data['device_model'].nunique(),
        'unique_locations': user_data['location_city'].nunique(),
        'avg_trust_score': round(user_data['trust_index'].mean(), 2),
        'min_trust_score': int(user_data['trust_index'].min()),
        'max_trust_score': int(user_data['trust_index'].max()),
        'high_risk_sessions': int((user_data['trust_index'] < 50).sum()),
        'low_risk_sessions': int((user_data['trust_index'] >= 90).sum()),
        'first_login': user_data['timestamp'].min(),
        'last_login': user_data['timestamp'].max()
    }


def visualize_confusion_matrix(cm: np.ndarray, output_path: str = None) -> None:
    """
    Create and display a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array
        output_path: Optional path to save the figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Anomaly', 'Normal'],
                yticklabels=['Anomaly', 'Normal'],
                cbar=True)
    plt.title('Confusion Matrix: Isolation Forest Anomaly Detection')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix plot saved to '{output_path}'")
    plt.close()


def plot_trust_score_distribution(df: pd.DataFrame, predictions: np.ndarray, output_path: str = None) -> None:
    """
    Visualize trust score distribution with anomaly highlighting.

    Args:
        df: DataFrame with trust_index column
        predictions: Model predictions (-1 for anomaly, 1 for normal)
        output_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Trust score distribution by prediction
    normal_scores = df.loc[predictions == 1, 'trust_index']
    anomaly_scores = df.loc[predictions == -1, 'trust_index']

    ax1.hist(normal_scores, bins=30, alpha=0.6, label='Normal (pred=1)', color='green')
    ax1.hist(anomaly_scores, bins=30, alpha=0.6, label='Anomaly (pred=-1)', color='red')
    ax1.set_xlabel('Trust Index Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Trust Score Distribution by Prediction')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Prediction breakdown
    pred_counts = pd.Series(predictions).value_counts()
    colors = ['red' if x == -1 else 'green' for x in pred_counts.index]
    labels = ['Anomaly' if x == -1 else 'Normal' for x in pred_counts.index]
    ax2.pie(pred_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Model Predictions Distribution')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Trust score distribution plot saved to '{output_path}'")
    plt.close()


def plot_feature_importance_by_risk(df: pd.DataFrame, predictions: np.ndarray, output_path: str = None) -> None:
    """
    Compare feature statistics between anomalies and normal behavior.

    Args:
        df: DataFrame with features
        predictions: Model predictions (-1 for anomaly, 1 for normal)
        output_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    normal_data = df[predictions == 1]
    anomaly_data = df[predictions == -1]

    # Hour comparison
    axes[0, 0].hist([normal_data['hour'], anomaly_data['hour']], bins=24,
                     label=['Normal', 'Anomaly'], alpha=0.6)
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Login Hour Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Trust Index comparison
    axes[0, 1].hist([normal_data['trust_index'], anomaly_data['trust_index']], bins=30,
                     label=['Normal', 'Anomaly'], alpha=0.6)
    axes[0, 1].set_xlabel('Trust Index Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Trust Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Device model comparison (unique count per group)
    axes[1, 0].bar(['Normal', 'Anomaly'],
                    [normal_data['device_model_encoded'].nunique(),
                     anomaly_data['device_model_encoded'].nunique()],
                    color=['green', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel('Unique Device Models')
    axes[1, 0].set_title('Device Model Diversity')
    axes[1, 0].grid(alpha=0.3, axis='y')

    # Location comparison (unique count per group)
    axes[1, 1].bar(['Normal', 'Anomaly'],
                    [normal_data['location_city_encoded'].nunique(),
                     anomaly_data['location_city_encoded'].nunique()],
                    color=['green', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel('Unique Locations')
    axes[1, 1].set_title('Location Diversity')
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to '{output_path}'")
    plt.close()


def generate_anomaly_detection_summary(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """
    Generate summary statistics comparing anomalies vs normal behavior.

    Args:
        df: DataFrame with all features
        predictions: Model predictions

    Returns:
        Summary statistics DataFrame
    """
    normal_data = df[predictions == 1]
    anomaly_data = df[predictions == -1]

    summary = pd.DataFrame({
        'Metric': [
            'Session Count',
            'Avg Trust Index',
            'Min Trust Index',
            'Max Trust Index',
            'Avg Hour (UTC)',
            'Unique Devices',
            'Unique Locations'
        ],
        'Normal Behavior': [
            len(normal_data),
            f"{normal_data['trust_index'].mean():.2f}",
            f"{normal_data['trust_index'].min():.0f}",
            f"{normal_data['trust_index'].max():.0f}",
            f"{normal_data['hour'].mean():.2f}",
            normal_data['device_model_encoded'].nunique(),
            normal_data['location_city_encoded'].nunique()
        ],
        'Anomalous Behavior': [
            len(anomaly_data),
            f"{anomaly_data['trust_index'].mean():.2f}" if len(anomaly_data) > 0 else 'N/A',
            f"{anomaly_data['trust_index'].min():.0f}" if len(anomaly_data) > 0 else 'N/A',
            f"{anomaly_data['trust_index'].max():.0f}" if len(anomaly_data) > 0 else 'N/A',
            f"{anomaly_data['hour'].mean():.2f}" if len(anomaly_data) > 0 else 'N/A',
            anomaly_data['device_model_encoded'].nunique() if len(anomaly_data) > 0 else 0,
            anomaly_data['location_city_encoded'].nunique() if len(anomaly_data) > 0 else 0
        ]
    })

    return summary

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import accuracy_score, r2_score

# CSV File path to read
csv_path = 'D:\Project\exercise_dataset.csv'

# Perform initial data exploration
def explore_data(csv_path):
    data = pd.read_csv(csv_path)
    print(data)
    print(data.dtypes)
    print(data.info)
    print(data.describe)
    print(data.head())
    print(data.tail())
    return data

# Function to visualize exercise counts
def visualize_exercise_counts(data):
    plt.figure(figsize=(8, 6))
    sorted_exercise_order = data['Exercise'].value_counts().index
    ax2 = sns.countplot(data=data, x='Exercise', order=sorted_exercise_order)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=14)
    ax2.set_title("Exercise used by Count")
    for p in ax2.patches:
        ax2.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.show()

# Function to visualize gender counts
def visualize_gender_counts(data):
    plt.figure(figsize=(8, 6))
    ax1 = sns.countplot(data=data, x='Gender')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=14)
    ax1.set_title("Gender used by Count")
    for p in ax1.patches:
        ax1.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()

# Function to visualize weather counts
def visualize_weather_counts(data):
    plt.figure(figsize=(8, 6))
    ax3 = sns.countplot(data=data, x='Weather Conditions')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=14)
    ax3.set_title("Weather used by Count")
    for p in ax3.patches:
        ax3.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()

# Function to visualize combinations of categorical columns
def visualize_categorical_combinations(data, categorical_columns):
    for i in range(len(categorical_columns)):
        for j in range(i+1, len(categorical_columns)):
            plt.figure(figsize=(8, 6))
            ax = sns.countplot(data=data, x=categorical_columns[i], hue=categorical_columns[j])
            plt.xlabel(categorical_columns[i])
            plt.ylabel("Count")
            plt.legend(title=categorical_columns[j])
            plt.title(f"{categorical_columns[i]} vs {categorical_columns[j]}")
            plt.xticks(rotation=45, ha="right")
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha='center', va='center', xytext=(0, 5), textcoords='offset points')
            plt.tight_layout()
            plt.show()

# Function to visualize histograms
def visualize_histogram(data, column_name):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x=column_name, bins=20, kde=True)
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column_name}')
    plt.show()

# Function to calculate weight status
def calculate_weight_status(data, threshold=1.0):
    data['Weight Difference'] = data['Actual Weight'] - data['Dream Weight']
    data['Weight Status'] = data['Weight Difference'].apply(
        lambda x: 'Weight Loss' if x > threshold else ('Weight Gain' if x < -threshold else 'Neutral')
    )
    return data

# Function to calculate additional exercise-related data
def calculate_exercise_data(data):
    exercise_data_dict = {}
    for exercise_name in data['Exercise'].unique():
        exercise_data = data[data['Exercise'] == exercise_name]
        total_calories_burned = exercise_data['Calories Burn'].sum()
        total_duration = exercise_data['Duration'].sum()
        calories_burned_per_minute = total_calories_burned / total_duration
        exercise_data_dict[exercise_name] = {'Total Calories Burned': total_calories_burned,
                                             'Total Duration': total_duration,
                                             'Calories Burned per Minute': calories_burned_per_minute}
    exercise_data = pd.DataFrame.from_dict(exercise_data_dict, orient='index')
    data = data.merge(exercise_data, left_on='Exercise', right_index=True)
    return data

# Function to create age groups
def create_age_groups(data):
    age_ranges = [0, 35, 55, float('inf')]
    age_labels = ['Young', 'Middle-aged', 'Old']
    data['Age Group'] = pd.cut(data['Age'], bins=age_ranges, labels=age_labels, right=False)
    return data

# Function to calculate height based on BMI
def calculate_height(data):
    data['Height'] = data.apply(lambda row: math.sqrt(row['Actual Weight'] / row['BMI']), axis=1)
    data['Height'] = data['Height'].round(2)
    return data

# Function to encode categorical variables
def encode_categorical_variables(data):
    label_encoder = LabelEncoder()
    data['Age Group Encoded'] = label_encoder.fit_transform(data['Age Group'])
    data['Gender Encoded'] = label_encoder.fit_transform(data['Gender'])
    data['Weather Encoded'] = label_encoder.fit_transform(data['Weather Conditions'])
    data['Exercise Encoded'] = label_encoder.fit_transform(data['Exercise'])
    return data

# Create correlation heatmap
def plot_correlation_heatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

# Function to prepare data for modeling (Exercise Prediction)
def prepare_data_exercise(data):
    X_columns = ['Dream Weight', 'Actual Weight', 'Calories Burned per Minute', 'Age Group Encoded', 'Gender Encoded', 'Height',
                 'BMI', 'Weather Encoded']
    y_column_exercise = 'Exercise Encoded'

    X = data[X_columns]
    y_exercise = data[y_column_exercise]

    X_train, X_test, y_train_exercise, y_test_exercise = train_test_split(X, y_exercise, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    noise_std = 0.0
    X_train_noisy = X_train + np.random.normal(scale=noise_std, size=X_train.shape)
    X_test_noisy = X_test + np.random.normal(scale=noise_std, size=X_test.shape)

    # Calculate and print accuracy for Exercise Prediction on train data
    model_exercise = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)
    model_exercise.fit(X_train_noisy, y_train_exercise)
    y_train_exercise_pred = model_exercise.predict(X_train_noisy)
    train_accuracy_exercise = accuracy_score(y_train_exercise, y_train_exercise_pred)
    print("Train Accuracy for Exercise Prediction:", train_accuracy_exercise)

    return X_train_noisy, X_test_noisy, y_train_exercise, y_test_exercise, scaler

# Function to prepare data for modeling (Duration Prediction)
def prepare_data_duration(data):
    X_columns = ['Dream Weight', 'Actual Weight', 'Weight Difference', 'Calories Burned per Minute', 'Age Group Encoded']
    y_column_duration = 'Duration'

    X = data[X_columns]
    y_duration = data[y_column_duration]

    X_train, X_test, y_train_duration, y_test_duration = train_test_split(X, y_duration, test_size=0.2, random_state=42)

    scaler_duration = StandardScaler()
    X_train = scaler_duration.fit_transform(X_train)
    X_test = scaler_duration.transform(X_test)

    noise_std = 0.0
    X_train_noisy_duration = X_train + np.random.normal(scale=noise_std, size=X_train.shape)
    X_test_noisy_duration = X_test + np.random.normal(scale=noise_std, size=X_test.shape)

    # Calculate and print R-squared for Duration Prediction on train data
    model_duration = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)
    model_duration.fit(X_train_noisy_duration, y_train_duration)
    y_train_duration_pred = model_duration.predict(X_train_noisy_duration)
    train_r2_duration = r2_score(y_train_duration, y_train_duration_pred)
    print("Train R-squared for Duration Prediction:", train_r2_duration)

    return X_train_noisy_duration, X_test_noisy_duration, y_train_duration, y_test_duration, scaler_duration



# Function to prepare data for modeling (Intensity Prediction)
def prepare_data_intensity(data):
    X_columns = ['Dream Weight', 'Actual Weight', 'Calories Burned per Minute', 'Age Group Encoded', 'Gender Encoded', 'Height',
                 'BMI', 'Weather Encoded']
    y_column_intensity = 'Exercise Intensity'

    X = data[X_columns]
    y_intensity = data[y_column_intensity]

    X_train, X_test, y_train_intensity, y_test_intensity = train_test_split(X, y_intensity, test_size=0.2, random_state=42)

    scaler_intensity = StandardScaler()
    X_train = scaler_intensity.fit_transform(X_train)
    X_test = scaler_intensity.transform(X_test)

    noise_std = 0.0
    X_train_noisy_intensity = X_train + np.random.normal(scale=noise_std, size=X_train.shape)
    X_test_noisy_intensity = X_test + np.random.normal(scale=noise_std, size=X_test.shape)

    # Calculate and print R-squared for Intensity Prediction on train data
    model_intensity = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)
    model_intensity.fit(X_train_noisy_intensity, y_train_intensity)
    y_train_intensity_pred = model_intensity.predict(X_train_noisy_intensity)
    train_r2_intensity = r2_score(y_train_intensity, y_train_intensity_pred)
    print("Train R-squared for Intensity Prediction:", train_r2_intensity)

    return X_train_noisy_intensity, X_test_noisy_intensity, y_train_intensity, y_test_intensity, scaler_intensity


# Predict Exercise, Duration, and Intensity for new data
def predict_all_new_data(new_data):
    new_data = pd.DataFrame(new_data)
    new_data['Weight Difference'] = new_data['Actual Weight'] - new_data['Dream Weight']  # Add this line
    scaler, scaler_duration, scaler_intensity = None, None, None
    model_exercise = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)
    model_duration = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)
    model_intensity = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)

    # Load and preprocess the data
    data = pd.read_csv(csv_path)
    data = calculate_weight_status(data)
    data = calculate_exercise_data(data)
    data = create_age_groups(data)
    data = calculate_height(data)
    data = encode_categorical_variables(data)

    # Prepare data for Exercise Prediction
    X_train_noisy, X_test_noisy, y_train_exercise, y_test_exercise, scaler = prepare_data_exercise(data)
    model_exercise.fit(X_train_noisy, y_train_exercise)
    

    # Prepare data for Duration Prediction
    X_train_noisy_duration, X_test_noisy_duration, y_train_duration, y_test_duration, scaler_duration = prepare_data_duration(data)
    model_duration.fit(X_train_noisy_duration, y_train_duration)

    # Prepare data for Intensity Prediction
    X_train_noisy_intensity, X_test_noisy_intensity, y_train_intensity, y_test_intensity, scaler_intensity = prepare_data_intensity(data)
    model_intensity.fit(X_train_noisy_intensity, y_train_intensity)

    # Predict Exercise for new data
    X_new = new_data[['Dream Weight', 'Actual Weight', 'Calories Burned per Minute', 'Age Group Encoded', 'Gender Encoded', 'Height', 'BMI', 'Weather Encoded']]
    X_new_scaled = scaler.transform(X_new)
    predictions_exercise = model_exercise.predict(X_new_scaled)

    # Predict Duration for new data
    X_new_duration = new_data[['Dream Weight', 'Actual Weight', 'Weight Difference', 'Calories Burned per Minute', 'Age Group Encoded']]
    X_new_scaled_duration = scaler_duration.transform(X_new_duration)
    predictions_duration = model_duration.predict(X_new_scaled_duration)

    # Predict Intensity for new data
    X_new_intensity = new_data[['Dream Weight', 'Actual Weight', 'Calories Burned per Minute', 'Age Group Encoded', 'Gender Encoded', 'Height', 'BMI', 'Weather Encoded']]
    X_new_scaled_intensity = scaler_intensity.transform(X_new_intensity)
    predictions_intensity = model_intensity.predict(X_new_scaled_intensity)

    return predictions_exercise, predictions_duration, predictions_intensity

if __name__ == "__main__":
    # Load and explore data
    data = explore_data(csv_path)

    # Visualize exercise counts
    visualize_exercise_counts(data)

    # Visualize gender counts
    visualize_gender_counts(data)

    # Visualize weather counts
    visualize_weather_counts(data)

    # Visualize combinations of categorical columns
    categorical_columns = ['Exercise', 'Gender', 'Weather Conditions']
    visualize_categorical_combinations(data, categorical_columns)
    
    # Visualize Correlation
    plot_correlation_heatmap(data)

    # Visualize histograms
    visualize_histogram(data, 'Calories Burn')
    
    visualize_histogram(data, 'Dream Weight')
    
    visualize_histogram(data, 'Actual Weight')
    
    visualize_histogram(data, 'Age')
    
    visualize_histogram(data, 'Duration')
    
    visualize_histogram(data, 'Heart Rate')
    
    visualize_histogram(data, 'Exercise Intensity')
    
        # Predict exercise, duration, and intensity for new data
    new_data = pd.DataFrame({
    'Dream Weight': [80, 68, 72, 85, 78, 73, 69, 74],
    'Actual Weight': [75, 71, 68, 88, 80, 70, 68, 72],
    'Calories Burned per Minute': [7.6, 7.0, 7.2, 7.5, 7.8, 7.3, 7.2, 7.1],
    'Age Group Encoded': [45, 25, 32, 55, 40, 28, 27, 35],
    'Gender Encoded': [0, 1, 0, 1, 0, 1, 0, 1],
    'Height': [178, 165, 170, 180, 175, 168, 167, 173],
    'BMI': [23.67, 26.04, 23.53, 27.16, 26.12, 24.8, 24.45, 24.18],
    'Weather Encoded': [2, 0, 1, 2, 0, 1, 2, 0]
})

    predictions_exercise, predictions_duration, predictions_intensity = predict_all_new_data(new_data)
    rounded_predictions_duration = np.round(predictions_duration)
    rounded_predictions_intensity = np.round(predictions_intensity)

    print("Predicted Exercise for new data:")
    print(predictions_exercise)

    print("Predicted Duration for new data:")
    print(rounded_predictions_duration)

    print("Predicted Intensity for new data:")
    print(rounded_predictions_intensity)
    
"----------------------------------------------------------------------------"
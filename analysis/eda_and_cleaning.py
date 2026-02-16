import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories for outputs
os.makedirs('analysis/plots', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded. Shape: {df.shape}")
    return df

def basic_inspection(df):
    print("\n--- Basic Inspection ---")
    print(df.info())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Duplicates ---")
    print(f"Duplicates found: {df.duplicated().sum()}")
    
    # Drop duplicates if any (Optional, but good practice if exact dupes)
    # df = df.drop_duplicates() 
    return df

def clean_data(df):
    print("\n--- Cleaning Data ---")
    
    # 1. Handle 'Sleep Disorder' NaN -> 'None'
    if 'Sleep Disorder' in df.columns:
        df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
        print("Filled NaN in 'Sleep Disorder' with 'None'")
        
    # 2. Split 'Blood Pressure' into Systolic and Diastolic
    if 'Blood Pressure' in df.columns:
        df[['BP_Systolic', 'BP_Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
        # Drop original column
        df = df.drop(columns=['Blood Pressure'])
        print("Split 'Blood Pressure' into 'BP_Systolic' and 'BP_Diastolic'")

    # 3. Normalize 'BMI Category'
    # 'Normal Weight' and 'Normal' seem to be the same
    if 'BMI Category' in df.columns:
        df['BMI Category'] = df['BMI Category'].replace({'Normal Weight': 'Normal'})
        print("Normalized 'BMI Category': Merged 'Normal Weight' into 'Normal'")

    return df

def generate_visualizations(df):
    print("\n--- Generating Visualizations ---")
    sns.set_style("whitegrid")
    
    # 1. Target Variable Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Sleep Disorder', data=df, palette='viridis', order=['None', 'Insomnia', 'Sleep Apnea'])
    plt.title('Distribution of Sleep Disorders')
    plt.savefig('analysis/plots/01_sleep_disorder_dist.png')
    plt.close()
    
    # 2. Correlation Heatmap (Numerical)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('analysis/plots/02_correlation_heatmap.png')
    plt.close()
    
    # 3. Sleep Duration by Disorder
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Sleep Disorder', y='Sleep Duration', data=df, palette='Set2')
    plt.title('Sleep Duration vs Sleep Disorder')
    plt.savefig('analysis/plots/03_sleep_duration_boxplot.png')
    plt.close()
    
    # 4. Stress Level by Disorder
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Sleep Disorder', y='Stress Level', data=df, palette='Set2')
    plt.title('Stress Level vs Sleep Disorder')
    plt.savefig('analysis/plots/04_stress_level_boxplot.png')
    plt.close()
    
    # 5. BMI Category Breakdown
    plt.figure(figsize=(10, 6))
    sns.countplot(x='BMI Category', hue='Sleep Disorder', data=df, palette='muted')
    plt.title('Sleep Disorder Distribution by BMI Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis/plots/05_bmi_distribution.png')
    plt.close()
    
    print("Plots saved in 'analysis/plots/'")

def main():
    # Path assumption based on user context
    data_path = 'data/Sleep_health_and_lifestyle_dataset.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        # Try finding it in the current directory or relative paths
        return

    df = load_data(data_path)
    df = basic_inspection(df)
    df_clean = clean_data(df)
    
    # Save cleaned data
    output_path = 'data/processed/sleep_health_cleaned.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to {output_path}")
    
    generate_visualizations(df_clean)
    
    print("\n--- Summary Stats of Cleaned Data ---")
    print(df_clean.describe())

if __name__ == "__main__":
    main()

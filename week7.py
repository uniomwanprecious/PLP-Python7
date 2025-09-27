# =================================================================
# Required Libraries
# =================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set a professional plotting style (as suggested in the instructions)
sns.set_theme()

# =================================================================
# Task 1: Load and Explore the Dataset
# =================================================================

print("--- Task 1: Load and Explore the Dataset ---")

# 1 & 2. Load the dataset (Using the suggested Iris dataset)
try:
    # Load Iris data directly into a DataFrame
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully (Iris).")

except Exception as e:
    # 4. Error Handling: Catch potential errors during loading
    print(f"Error loading dataset: {e}")
    # A boss-level exit if the core data can't be loaded
    exit()

# 3. Display the first few rows
print("\nFirst 5 rows of the dataset (df.head()):")
print(df.head())

# 4. Explore the structure (data types and missing values)
print("\nDataset structure (df.info()):")
# .info() returns None, so we print the result of the call
df.info()

# 5. Clean the dataset
# For the Iris dataset, cleaning is often not strictly necessary, 
# but a check for nulls is good practice.

if df.isnull().sum().sum() > 0:
    print("\nMissing values detected. Cleaning data...")
    # Example cleaning strategy: Drop rows with any missing data
    df_cleaned = df.dropna()
    print(f"Dropped {len(df) - len(df_cleaned)} rows with missing data.")
    df = df_cleaned
else:
    print("\nNo missing values found. Data is clean.")


# =================================================================
# Task 2: Basic Data Analysis
# =================================================================

print("\n--- Task 2: Basic Data Analysis ---")

# 1. Compute the basic statistics of the numerical columns
print("\nBasic Statistics of Numerical Columns (df.describe()):")
print(df.describe())

# 2. Perform groupings on a categorical column and compute the mean
# Group by 'species' and calculate the mean of 'petal length (cm)'
grouped_means = df.groupby('species')['petal length (cm)'].mean()
print("\nMean Petal Length by Species (Groupings):")
print(grouped_means)

# 3. Identify any patterns or interesting findings (Interpretation)
print("\nPatterns and Findings:")
print(" - Finding 1 (from describe): The high standard deviation for 'petal length (cm)' (approx. 1.76) suggests a large variance in petal sizes compared to 'sepal width (cm)' (approx. 0.43).")
print(" - Finding 2 (from groupings): There is a clear separation in mean petal length between the species, with 'setosa' having the smallest (approx. 1.46) and 'virginica' having the largest (approx. 5.55).")


# =================================================================
# Task 3: Data Visualization
# =================================================================

print("\n--- Task 3: Data Visualization ---")
plt.figure(figsize=(12, 10))

# 1. Line chart (Placeholder/Example - Iris data is not time-series, so we plot an index-based trend for illustration)
plt.subplot(2, 2, 1) # 2 rows, 2 columns, 1st plot
df['petal length (cm)'].plot(kind='line', title='Trends in Petal Length (Index-based)')
plt.xlabel('Data Point Index')
plt.ylabel('Petal Length (cm)')

# 2. Bar chart (Comparison of a numerical value across categories)
plt.subplot(2, 2, 2) # 2 rows, 2 columns, 2nd plot
# Use the grouped means calculated earlier
sns.barplot(x=grouped_means.index, y=grouped_means.values)
plt.title('Average Petal Length by Species')
plt.xlabel('Flower Species')
plt.ylabel('Mean Petal Length (cm)')

# 3. Histogram (Distribution of a numerical column)
plt.subplot(2, 2, 3) # 2 rows, 2 columns, 3rd plot
plt.hist(df['sepal width (cm)'], bins=10, edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')

# 4. Scatter plot (Relationship between two numerical columns)
plt.subplot(2, 2, 4) # 2 rows, 2 columns, 4th plot
# Use 'hue' to show the relationship relative to the categorical variable
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Relationship: Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Display all plots
plt.show()

print("\n--- Script Complete ---")
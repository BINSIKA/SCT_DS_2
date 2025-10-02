import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\binsi\OneDrive\Desktop\Project\skillcraft Technology\Titanic-Dataset.csv")

# Set up the visualization style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

## --- Visualization 1: Count Plot of Survival ---
plt.subplot(3, 2, 1) # 3 rows, 2 columns, 1st plot
sns.countplot(x='Survived', data=df)
plt.title('1. Survival Count (0=No, 1=Yes)', fontsize=12)
plt.xlabel('Survived', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.xticks([0, 1], ['Did Not Survive', 'Survived'])

## --- Visualization 2: Survival Rate by Passenger Class (Pclass) ---
plt.subplot(3, 2, 2)
sns.barplot(x='Pclass', y='Survived', data=df, palette='viridis')
plt.title('2. Survival Rate by Passenger Class', fontsize=12)
plt.xlabel('Passenger Class (1st, 2nd, 3rd)', fontsize=10)
plt.ylabel('Survival Rate', fontsize=10)
plt.xticks([0, 1, 2], ['1st Class', '2nd Class', '3rd Class'])

## --- Visualization 3: Survival Count by Sex ---
plt.subplot(3, 2, 3)
sns.countplot(x='Sex', hue='Survived', data=df, palette='coolwarm')
plt.title('3. Survival Count by Sex', fontsize=12)
plt.xlabel('Sex', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.legend(title='Survived', labels=['Did Not Survive', 'Survived'])

## --- Visualization 4: Distribution of Age ---
plt.subplot(3, 2, 4)
sns.histplot(df['Age'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('4. Age Distribution', fontsize=12)
plt.xlabel('Age', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

## --- Visualization 5: Survival Rate by Port of Embarkation (Embarked) ---
# Fill missing Embarked values with the mode 'S' for visualization
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
plt.subplot(3, 2, 5)
sns.barplot(x='Embarked', y='Survived', data=df, palette='magma')
plt.title('5. Survival Rate by Port of Embarkation', fontsize=12)
plt.xlabel('Port (C=Cherbourg, Q=Queenstown, S=Southampton)', fontsize=10)
plt.ylabel('Survival Rate', fontsize=10)

## --- Visualization 6: Fare Distribution (Box Plot) ---
plt.subplot(3, 2, 6)
sns.boxplot(y='Fare', data=df, color='lightcoral')
plt.title('6. Fare Distribution', fontsize=12)
plt.ylabel('Fare', fontsize=10)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-axis ticks/labels

# Display the plots
plt.tight_layout()
plt.savefig("titanic dataset") # Adjust subplot params for tight layout
plt.show()
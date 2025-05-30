import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("titanic.csv")  # Make sure this file is in the same folder

# 2. Basic info
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Null Values ---")
print(df.isnull().sum())

# 3. Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 4. Convert categorical data
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 5. Normalize numeric features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 6. Visualize outliers
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.show()

# 7. Remove outliers using IQR
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

# 8. Save cleaned data
df.to_csv("titanic_cleaned.csv", index=False)

print("\n✅ Preprocessing Complete. Cleaned file saved as 'titanic_cleaned.csv'")

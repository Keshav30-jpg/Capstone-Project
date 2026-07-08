import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import null


### First Task: Load the CSV file and check the head, data types and shape of the row and columns

df = pd.read_csv("churnguard_data.csv")
print("First 5 rows:\n", df.head())
print("\nData Types:\n", df.dtypes)
print("\nShape:", df.shape)

### Second Task: now we cheack the Null Values.

null_counts = df.isnull().sum()
null_percentages = (null_counts / df.shape[0]) * 100
null_table = pd.DataFrame({'Counts': null_counts, 'Percentage': null_percentages})
print("\nNull value Table:\n", null_table)

### we need to indentify the the columns above 20 percentage null rate and also fill the columns below 20 percentage nulls with median

high_null_cols = null_percentages[null_percentages > 20]. index.tolist()
print("Columns with >20% nulls:", high_null_cols)

### in my CSV File raw dataset i don't have any columns with grater than 20 percentage, i will  give detailed explaination in README file
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if null_percentages[col] <= 20:
        df[col] = df[col].fillna(df[col].median())

#print("df.shape():", df.shape)
#print("First 66 rows:\n", df.head(66))


### Third Task: we need to remove the duplicates in the raw dataset
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Removed {duplicate_count} duplicate rows.")

#print(f"df.shape: {df.shape}")

### Fourth Task: we do the Type Conversion and memory usage
mem_before = df.memory_usage(deep=True).sum()
df.columns = df.columns.str.strip()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["PhoneService"] = df["PhoneService"].astype(str).str.lower().str.strip()
df["InternetService"] = df["InternetService"].astype(str).str.lower().str.strip()
internet_mapping = {"fiberoptic": "fiber optic", "fiber optic": "fiber optic", "fibre optic": "fiber optic", "nan": "no"}
df['InternetService'] = df['InternetService'].replace(internet_mapping)
df['Contract'] = df['Contract'].astype(str).str.lower().str.strip()
contract_mapping ={ "1 year": "one year", "2 year": "two year", "month to month": "month-to-month"}
df['Contract'] = df['Contract'].replace(contract_mapping)
df['PaperlessBilling'] = df['PaperlessBilling'].astype(str).str.lower().str.strip()
df['Churn'] = df['Churn'].astype(str).str.lower().str.strip()

category_cols = ["PhoneService", "InternetService", "Contract", "PaperlessBilling", "Churn"]
for col in category_cols:
    df[col] = df[col].astype("category")

mem_after = df.memory_usage(deep=True).sum()
print(f"Memory Usage Before: {mem_before} bytes")
print(f"Memory Usage After: {mem_after} bytes")

#print("First 66 rows:\n", df.head(66))  #cross check
### Fifth Task: skewness 

print(df.describe())
print(f"Original MonthlyCharges Skewness: {df['MonthlyCharges'].skew():.4f}")
print(f"Original tenure Skewness: {df['tenure'].skew():.4f}")
df.loc[df['tenure'] < 0, 'tenure'] = df['tenure'].median()
df.loc[df['MonthlyCharges'] > 200, 'MonthlyCharges'] = df['MonthlyCharges'].median()


print(df.describe())
all_numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
skews = {}
for col in all_numeric:
    skews[col] = df[col].skew()
    print(f"Skewness of {col}: {skews[col]:.4f}")
highest_skew_col = max(skews, key=lambda k: abs(skews[k]))
print(f"Column with the highest absolute skewness: {highest_skew_col}")

#while i doing it i check the output the Total Charges and Monthly Charges column the max is goes up to higher Value and the skewness and highily chances for outlier
#df.loc[df['tenure'] < 0, 'tenure'] = df['tenure'].median()
#df.loc[df['MonthlyCharges'] > 200, 'MonthlyCharges'] = df['MonthlyCharges'].median()        so i add this two lines before the current df.describe() 

### sixth Task: Outlier check with IQR
for col in ["MonthlyCharges", "TotalCharges"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    print(f"{col} Outliers detected: {outliers_count} (Bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")



###  seventh Task: for Visualization it automatically create a separate .png file all visualization
### line plot

plt.figure(figsize=(8, 4))
plt.plot(df.index[:100], df["MonthlyCharges"].iloc[:100], color="blue", alpha=0.7)
plt.title("Line Plot: Monthly Charges Trend (First 100 Rows)")
plt.xlabel("Row Index")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("line_plot.png")  
plt.close()

### bar plot

plt.figure(figsize=(8,4))
df.groupby("Contract")["MonthlyCharges"].mean().plot(kind="bar", color="orange", edgecolor="black")
plt.title("Average Monthly charges by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Mean Monthly Charges")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("bar_plot.png")
plt.close()

### histogram

plt.figure(figsize=(8, 4))
sns.histplot(df[highest_skew_col], bins=20, kde=True, color='purple')
plt.title(f"Histogram: Distribution of {highest_skew_col}")
plt.xlabel(highest_skew_col)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("histogram.png")
plt.close()

### scatter plot

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="tenure", y="TotalCharges", hue="Churn", alpha=0.6)
plt.title("Scatter Plot: Tenure vs Total Charges")
plt.xlabel("Tenure (Months)")
plt.ylabel("Total Charges")
plt.tight_layout()
plt.savefig("scatter_plot.png")  
plt.close()

### box plot

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="InternetService", y="MonthlyCharges", palette="Set2")
plt.title("Box Plot: Monthly Charges Spread across Internet Services")
plt.xlabel("Internet Service Type")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("box_plot.png")  
plt.close()
### eighth Task:
### heatmap

plt.figure(figsize=(6, 5))
pearson_matrix = df[all_numeric].corr(method="pearson")
sns.heatmap(pearson_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1)
plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")  
plt.close()

### ninth Task: 
### a) Imputation strategy comparison get a two columns and compare with mean and median
sorted_skews = sorted(skews.items(), key=lambda x: abs(x[1]), reverse=True)
top_2_skewed = [sorted_skews[0][0], sorted_skews[1][0]]
for col in top_2_skewed:
    print(f"Column: {col}")
    print(f"  Mean:   {df[col].mean():.4f}")
    print(f"  Median: {df[col].median():.4f}")

### b) Spearman rank correlation
spearman_matrix = df[all_numeric].corr(method="spearman")
print("Spearman Matrix:\n", spearman_matrix)
difference_table = (spearman_matrix - pearson_matrix).abs()
print("\nAbsolute Difference Table (|Spearman - Pearson|):\n", difference_table)

### c) Grouped aggregation
grouped_df = df.groupby("InternetService")["MonthlyCharges"].agg(["mean", "std", "count"])
print(grouped_df)

### tenth Task: Save the clean dataset
df.to_csv('cleaned_data.csv', index=False)
print("\nSuccessfully saved 'cleaned_data.csv' to root directory!")

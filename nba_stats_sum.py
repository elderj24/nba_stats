# NBA Salary Analysis (Summary Version): How do advanced performance metrics relate to player salaries?
# Refer to public github repo: [  ] for further exploration and analysis 
# Data source: basketball-reference.com (refer to nba_scraping.ipynb for data scraping details)

#%%
#1 Import libraries and load csv data
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df_stats = pd.read_csv('nba_advanced_stats_2025.csv')
df_salary = pd.read_csv('nba_player_salaries_2025.csv')

#%%
#2 Data Cleaning
# Salary Data: Remove nulls from page breaks and any duplicates (take first entry)
df_salary_updated = df_salary.dropna(subset=['Salary_2025_26']).drop_duplicates(subset=['Player'], keep='first')

# Stats Data: Clean up players with multiple entries due to mid-season trades 
# Identify players with 2TM/3TM entries (signifying multiple teams)
has_combined = df_stats[df_stats['Team'].isin(['2TM', '3TM'])]['Player'].unique()

# Keep only 2TM/3TM rows for traded players, and all rows for non-traded players
df_stats_updated = df_stats[
    (df_stats['Player'].isin(has_combined) & df_stats['Team'].isin(['2TM', '3TM'])) |
    (~df_stats['Player'].isin(has_combined))
].reset_index(drop=True)

#%%
# Merge datasets and create salary in millions
df_merged = pd.merge(df_stats_updated, df_salary_updated[['Player', 'Salary_2025_26']], on='Player', how='inner')
df_merged['Salary_Millions'] = df_merged['Salary_2025_26'] / 1_000_000

#%%
# Examine a correlation matrix between the variables of interest
correlation_matrix = df_merged[['PER', 'WS', 'VORP', 'Salary_Millions']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Performance Metrics vs Salary', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

#%%
# Create a Multi-Linear Regression Model to forecast salary based on select advanced stats
features_to_use = ['PER', 'TS%', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP', 
            'MP', 'G', 'USG%', 'AST%', 'TRB%']

df_model = df_merged[features_to_use + ['Salary_Millions']].dropna()
X = df_model[features_to_use]
y = df_model['Salary_Millions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)

# Model Evaluation
print(f"\nModel Performance (Test Set):")
print(f"  R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}M")
print(f"  MAE: ${mean_absolute_error(y_test, y_pred_test):.2f}M")

#%%
# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Salary ($ Millions)')
plt.ylabel('Predicted Salary ($ Millions)')
plt.title('Predicted vs Actual Salaries (Test Set)')
plt.tight_layout()
plt.show()

#%%
# Identify Over/Underpaid Players
results_df = pd.DataFrame({
    'Player': df_merged.loc[X_test.index, 'Player'].values,
    'Age': df_merged.loc[X_test.index, 'Age'].values,
    'Actual': y_test.values,
    'Predicted': y_pred_test,
    'Difference': y_test.values - y_pred_test
}).sort_values('Difference', ascending=False)

print("TOP 3 OVERPAID PLAYERS (Actual > Predicted)")
print(results_df.head(3).to_string(index=False))

print("TOP 3 UNDERPAID PLAYERS (Predicted > Actual)")
print(results_df.tail(3).to_string(index=False))

results_df.to_csv('salary_predictions_sum.csv', index=False)
print("\nFull results saved to 'salary_predictions.csv'")

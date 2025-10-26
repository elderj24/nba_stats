# NBA Advanced Player Stats and Salary Scraper - Jupyter Notebook Style
# Run each cell step by step

# %%
# Cell 1: Import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# %%
# Cell 1A: Load existing data if available
# Check if we already have the full dataset saved
if os.path.exists('nba_advanced_stats_with_salary_2025.csv'):
    print("Found existing data file! Loading...")
    df_merged = pd.read_csv('nba_advanced_stats_with_salary_2025.csv')
    df_merged_qualified = df_merged[df_merged['MP'] >= 15].copy()
    
    # Create df_for_ranking for analysis cells
    df_for_ranking = df_merged_qualified.copy()
    df_for_ranking['is_tot'] = df_for_ranking['Team'] == 'TOT'
    df_for_ranking = df_for_ranking.sort_values('is_tot', ascending=False).drop_duplicates(subset='Player', keep='first')
    
    print(f"Loaded {len(df_merged)} stat rows")
    print(f"Qualified players (15+ MPG): {len(df_merged_qualified)}")
    print(f"Unique qualified players for ranking: {len(df_for_ranking)}")
    print("\nData ready! You can now skip to Cell 21 to view analysis!")
    print("Or continue from Cell 2 to re-scrape fresh data.")
else:
    print("No existing data found. Run cells 2+ to scrape data.")

# %%
# Cell 2: Set up the URL and headers for advanced stats
url = 'https://www.basketball-reference.com/leagues/NBA_2025_advanced.html'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}
# %%
# Cell 3: Make the request for advanced stats
print(f"Fetching data from {url}...")
response = requests.get(url, headers=headers)
print(f"Status code: {response.status_code}")

# %%
# Cell 4: Parse the HTML
soup = BeautifulSoup(response.content, 'lxml')

# The advanced stats table likely has id='advanced'
table = soup.find('table', {'id': 'advanced'})

if not table:
    # Sometimes it's in the page differently, let's check all tables
    all_tables = soup.find_all('table')
    print(f"Found {len(all_tables)} tables")
    for t in all_tables:
        print(f"Table ID: {t.get('id')}")
    table = all_tables[0] if all_tables else None

print("Table found!" if table else "Table not found")

# %%
# Cell 5: Extract column headers
headers_list = []
for th in table.find('thead').find_all('th'):
    headers_list.append(th.text.strip())
print(f"Columns: {headers_list}")

# %%
# Cell 6: Extract all rows
rows = []
for tr in table.find('tbody').find_all('tr'):
    # Skip header rows that appear in the middle
    if tr.find('th', {'scope': 'row'}) is None:
        continue
    
    row = []
    for td in tr.find_all(['th', 'td']):
        row.append(td.text.strip())
    
    if row:
        rows.append(row)

print(f"Scraped {len(rows)} player records")

# %%
# Cell 7: Create DataFrame
df = pd.DataFrame(rows, columns=headers_list)
print(df.head())

# %%
# Cell 8: Convert numeric columns
# Advanced stats are mostly numeric after the first few columns
numeric_cols = df.columns[5:]  # Stats start after Pos column
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nData types after conversion:")
print(df.dtypes)

# %%
# Cell 9: Explore the advanced stats
print(f"\nTotal players: {len(df)}")
print(f"\nKey Advanced Stats columns:")
print(df[['Player', 'Team', 'PER', 'TS%', 'WS', 'BPM']].describe())

# %%
# Cell 10: Top 10 by PER (Basketball Reference's official PER)
print("\nTop 10 Players by PER:")
top_per = df.nlargest(10, 'PER')[['Player', 'Team', 'PER', 'TS%', 'WS', 'BPM']]
print(top_per)

# %%
# Cell 11: Top 10 by Win Shares
print("\nTop 10 Players by Win Shares:")
top_ws = df.nlargest(10, 'WS')[['Player', 'Team', 'MP', 'PER', 'WS', 'WS/48']]
print(top_ws)

# %%
# Cell 12: Filter for qualified players (at least 15 MPG)
df_qualified = df[df['MP'] >= 15].copy()

print(f"\nQualified players (15+ MPG): {len(df_qualified)}")
print(f"Average PER for qualified: {df_qualified['PER'].mean():.2f}")

print("\nTop 10 Qualified Players by PER:")
top_per_qual = df_qualified.nlargest(10, 'PER')[['Player', 'Team', 'MP', 'PER', 'TS%', 'WS', 'BPM']]
print(top_per_qual)

# %%
# Cell 13: Save advanced stats to CSV
df.to_csv('nba_advanced_stats_2025.csv', index=False)
print("\nAdvanced stats saved to nba_advanced_stats_2025.csv")

# %%
# Cell 14: Scrape player salaries
salary_url = 'https://www.basketball-reference.com/contracts/players.html'

print(f"\nFetching salary data from {salary_url}...")
salary_response = requests.get(salary_url, headers=headers)
print(f"Status code: {salary_response.status_code}")

# %%
# Cell 15: Parse salary table
salary_soup = BeautifulSoup(salary_response.content, 'lxml')
salary_table = salary_soup.find('table', {'id': 'player-contracts'})
print("Salary table found!" if salary_table else "Salary table not found")

# %%
# Cell 16: Extract salary data with correct column names
correct_headers = ['Rk', 'Player', 'Tm', '2025-26', '2026-27', '2027-28', '2028-29', '2029-30', '2030-31', 'Guaranteed']

salary_rows = []
for tr in salary_table.find('tbody').find_all('tr'):
    row = []
    
    # Get all cells in the row
    cells = tr.find_all(['th', 'td'])
    
    for cell in cells:
        row.append(cell.text.strip())
    
    # Only keep rows that have 10 columns (matching our headers)
    if len(row) == 10:
        salary_rows.append(row)

print(f"Scraped {len(salary_rows)} salary records with 10 columns")

# %%
# Cell 17: Create salary DataFrame
df_salary = pd.DataFrame(salary_rows, columns=correct_headers)
print(f"\nSalary columns: {df_salary.columns.tolist()}")
print(f"Shape: {df_salary.shape}")

# %%
# Cell 18: Clean salary data
df_salary_clean = df_salary[['Player', '2025-26']].copy()

# Clean the salary column - ensure it's a string first
df_salary_clean['Salary_2025_26'] = df_salary_clean['2025-26'].astype(str).str.replace('$', '').str.replace(',', '')
df_salary_clean['Salary_2025_26'] = pd.to_numeric(df_salary_clean['Salary_2025_26'], errors='coerce')

# Drop the raw column
df_salary_clean = df_salary_clean[['Player', 'Salary_2025_26']].copy()

print(f"\nSalary data cleaned.")
print(f"Total players with salary: {len(df_salary_clean)}")
print(df_salary_clean.head(10))

# Create CSV file 
df_salary_clean.to_csv('nba_player_salaries_2025.csv', index=False)
print("\nSalary data saved to nba_player_salaries_2025.csv")    

# %%
# Cell 19: Merge salary with advanced stats
df_merged = df.merge(df_salary_clean, on='Player', how='left')

print(f"\nMerge complete!")
print(f"Total stat rows (including multi-team players): {len(df_merged)}")
print(f"Unique players: {df_merged['Player'].nunique()}")
print(f"Total rows with salary data: {df_merged['Salary_2025_26'].notna().sum()}")

# Create qualified dataset
df_merged_qualified = df_merged[df_merged['MP'] >= 15].copy()

print(f"\nQualified players (15+ MPG):")
print(f"Total qualified rows: {len(df_merged_qualified)}")
print(f"Qualified rows with salary: {df_merged_qualified['Salary_2025_26'].notna().sum()}")

# %%
# Cell 20: Handle multi-team players - prefer TOT rows
df_for_ranking = df_merged_qualified.copy()

# Create a preference: TOT rows first, then single team rows
df_for_ranking['is_tot'] = df_for_ranking['Team'] == 'TOT'

# Remove duplicate players, keeping TOT if available
df_for_ranking = df_for_ranking.sort_values('is_tot', ascending=False).drop_duplicates(subset='Player', keep='first')

print(f"\nUnique qualified players for ranking: {len(df_for_ranking)}")

# %%
# Cell 21: Top performers by various advanced stats
print("\n" + "="*70)
print("TOP 10 PLAYERS BY PER (with salary)")
print("="*70)
top_per_salary = df_for_ranking.nlargest(10, 'PER')[['Player', 'Team', 'MP', 'PER', 'TS%', 'WS', 'Salary_2025_26']]
print(top_per_salary)

print("\n" + "="*70)
print("TOP 10 PLAYERS BY WIN SHARES")
print("="*70)
top_ws = df_for_ranking.nlargest(10, 'WS')[['Player', 'Team', 'MP', 'PER', 'WS', 'WS/48', 'Salary_2025_26']]
print(top_ws)

print("\n" + "="*70)
print("TOP 10 PLAYERS BY BPM (Box Plus/Minus)")
print("="*70)
top_bpm = df_for_ranking.nlargest(10, 'BPM')[['Player', 'Team', 'MP', 'PER', 'BPM', 'VORP', 'Salary_2025_26']]
print(top_bpm)

# %%
# Cell 22: Value analysis - PER per million
df_for_ranking['Value_PER'] = df_for_ranking['PER'] / (df_for_ranking['Salary_2025_26'] / 1_000_000)
df_for_ranking['Value_PER'] = df_for_ranking['Value_PER'].round(2)

# Also calculate WS per million
df_for_ranking['Value_WS'] = df_for_ranking['WS'] / (df_for_ranking['Salary_2025_26'] / 1_000_000)
df_for_ranking['Value_WS'] = df_for_ranking['Value_WS'].round(2)

# Filter for players making at least $5M
high_earners = df_for_ranking[df_for_ranking['Salary_2025_26'] >= 5_000_000].copy()

print("\n" + "="*70)
print("TOP 10 BEST VALUE PLAYERS - PER per $1M (min $5M salary)")
print("="*70)
top_value_per = high_earners.nlargest(10, 'Value_PER')[['Player', 'Team', 'PER', 'Salary_2025_26', 'Value_PER']]
print(top_value_per)

print("\n" + "="*70)
print("TOP 10 BEST VALUE PLAYERS - Win Shares per $1M (min $5M salary)")
print("="*70)
top_value_ws = high_earners.nlargest(10, 'Value_WS')[['Player', 'Team', 'WS', 'Salary_2025_26', 'Value_WS']]
print(top_value_ws)

print("\n" + "="*70)
print("MOST OVERPAID PLAYERS (min $20M salary, lowest PER)")
print("="*70)
overpaid = df_for_ranking[df_for_ranking['Salary_2025_26'] >= 20_000_000].nsmallest(10, 'PER')[['Player', 'Team', 'PER', 'WS', 'Salary_2025_26', 'Value_PER']]
print(overpaid)

# %%
# Cell 23: Efficiency rankings - True Shooting %
print("\n" + "="*70)
print("TOP 10 MOST EFFICIENT SCORERS (True Shooting %)")
print("="*70)
# Filter for players with meaningful scoring (at least 10 PPG in basic stats would be ideal, but we'll use qualified)
top_ts = df_for_ranking.nlargest(10, 'TS%')[['Player', 'Team', 'TS%', 'PER', 'Salary_2025_26']]
print(top_ts)

# %%
# Cell 24: Save merged data
df_merged.to_csv('nba_advanced_stats_with_salary_2025.csv', index=False)
print("\n" + "="*70)
print("Full data with salary saved to nba_advanced_stats_with_salary_2025.csv")
print("="*70)

# %%
# Cell 25: Correlation Analysis with Heatmap
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*70)
print("CORRELATION ANALYSIS - Advanced Stats vs Salary")
print("="*70)

# Prepare data - only use players with salary data
corr_data = df_for_ranking[df_for_ranking['Salary_2025_26'].notna()].copy()

# Select features for correlation analysis
corr_cols = ['Salary_2025_26', 'PER', 'TS%', 'WS', 'WS/48', 'BPM', 'VORP', 'MP', 'Age']

# Create correlation matrix
corr_matrix = corr_data[corr_cols].corr()

print("\nCorrelation with Salary:")
salary_corr = corr_matrix['Salary_2025_26'].sort_values(ascending=False)
print(salary_corr)

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Matrix: Advanced Stats vs Salary', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nHeatmap saved as 'correlation_heatmap.png'")

# %%
# Cell 26: Individual Scatter Plots - Top Correlations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Salary vs Advanced Stats - Scatter Plots', fontsize=16, fontweight='bold')

# Top stats to plot (excluding salary itself)
top_stats = ['PER', 'WS', 'VORP', 'BPM', 'WS/48', 'TS%']

for idx, stat in enumerate(top_stats):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Create scatter plot
    ax.scatter(corr_data[stat], corr_data['Salary_2025_26'] / 1_000_000, 
               alpha=0.6, s=50)
    
    # Add trend line
    z = np.polyfit(corr_data[stat].dropna(), 
                   (corr_data['Salary_2025_26'] / 1_000_000).dropna(), 1)
    p = np.poly1d(z)
    ax.plot(corr_data[stat].sort_values(), 
            p(corr_data[stat].sort_values()), 
            "r--", alpha=0.8, linewidth=2)
    
    # Labels and formatting
    ax.set_xlabel(stat, fontsize=11, fontweight='bold')
    ax.set_ylabel('Salary ($M)', fontsize=11, fontweight='bold')
    correlation = corr_matrix.loc[stat, 'Salary_2025_26']
    ax.set_title(f'{stat} (r = {correlation:.3f})', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('salary_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nScatter plots saved as 'salary_scatter_plots.png'")

# %%
# Cell 27: Linear Regression - Predict Salary from Advanced Stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

print("\n" + "="*70)
print("SALARY PREDICTION MODEL - Based on Advanced Stats")
print("="*70)

# Prepare data - only use players with salary data
model_data = df_for_ranking[df_for_ranking['Salary_2025_26'].notna()].copy()

# Select features (advanced stats that should correlate with salary)
feature_cols = ['PER', 'TS%', 'WS', 'WS/48', 'BPM', 'VORP', 'MP']

# Remove any rows with missing values in these columns
model_data = model_data[feature_cols + ['Salary_2025_26']].dropna()

print(f"\nTraining on {len(model_data)} players with complete data")

# Prepare X (features) and y (target)
X = model_data[feature_cols]
y = model_data['Salary_2025_26']

# Split into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\nModel Performance:")
print(f"Training R² Score: {train_r2:.3f}")
print(f"Test R² Score: {test_r2:.3f}")
print(f"Mean Absolute Error: ${test_mae:,.0f}")

# Show feature importance (coefficients)
print("\nFeature Importance (Coefficients):")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print(feature_importance)

print(f"\nIntercept: ${model.intercept_:,.0f}")

# %%
# Cell 28: Visualize Model Performance
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Actual vs Predicted
ax1 = axes[0]
ax1.scatter(y_test / 1_000_000, y_pred_test / 1_000_000, alpha=0.6, s=100)
ax1.plot([0, y_test.max() / 1_000_000], [0, y_test.max() / 1_000_000], 
         'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Salary ($M)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Salary ($M)', fontsize=12, fontweight='bold')
ax1.set_title(f'Actual vs Predicted Salary\nTest R² = {test_r2:.3f}', 
              fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[1]
residuals = (y_test - y_pred_test) / 1_000_000
ax2.scatter(y_pred_test / 1_000_000, residuals, alpha=0.6, s=100)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Salary ($M)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residual ($M)', fontsize=12, fontweight='bold')
ax2.set_title('Residual Plot\n(Actual - Predicted)', 
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nModel performance plots saved as 'model_performance.png'")

# %%
# Cell 29: Show biggest prediction errors (overpaid/underpaid vs model)
# Add predictions to the full dataset
model_data_full = df_for_ranking[df_for_ranking['Salary_2025_26'].notna()].copy()

# First, drop NaN values from feature columns only
model_data_full = model_data_full.dropna(subset=feature_cols)

# Now extract features and make predictions
X_full = model_data_full[feature_cols]
model_data_full['Predicted_Salary'] = model.predict(X_full)
model_data_full['Salary_Diff'] = model_data_full['Salary_2025_26'] - model_data_full['Predicted_Salary']
model_data_full['Salary_Diff_Pct'] = (model_data_full['Salary_Diff'] / model_data_full['Predicted_Salary'] * 100).round(1)

print("\n" + "="*70)
print("MOST UNDERPAID (Actual salary much lower than predicted)")
print("="*70)
underpaid = model_data_full.nsmallest(10, 'Salary_Diff')[
    ['Player', 'Team', 'PER', 'WS', 'Salary_2025_26', 'Predicted_Salary', 'Salary_Diff', 'Salary_Diff_Pct']
]
print(underpaid)

print("\n" + "="*70)
print("MOST OVERPAID (Actual salary much higher than predicted)")
print("="*70)
overpaid_model = model_data_full.nlargest(10, 'Salary_Diff')[
    ['Player', 'Team', 'PER', 'WS', 'Salary_2025_26', 'Predicted_Salary', 'Salary_Diff', 'Salary_Diff_Pct']
]
print(overpaid_model)

print("\n" + "="*70)
print("MOST ACCURATE PREDICTIONS (Model got these right)")
print("="*70)
model_data_full['Abs_Diff'] = abs(model_data_full['Salary_Diff'])
accurate = model_data_full.nsmallest(10, 'Abs_Diff')[
    ['Player', 'Team', 'PER', 'WS', 'Salary_2025_26', 'Predicted_Salary', 'Salary_Diff_Pct']
]
print(accurate)

# %%
# Cell 30: Distribution Plots for Advanced Stats and Salaries
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribution of Advanced Stats and Salaries (Qualified Players)', 
             fontsize=16, fontweight='bold')

# Data for plotting - only qualified players with salary
plot_data = df_for_ranking[df_for_ranking['Salary_2025_26'].notna()].copy()

# Stats to plot
stats_to_plot = [
    ('PER', 'Player Efficiency Rating', 'skyblue'),
    ('WS', 'Win Shares', 'lightcoral'),
    ('VORP', 'Value Over Replacement', 'lightgreen'),
    ('BPM', 'Box Plus/Minus', 'plum'),
    ('TS%', 'True Shooting %', 'peachpuff'),
    ('Salary_2025_26', 'Salary ($M)', 'gold')
]

for idx, (stat, title, color) in enumerate(stats_to_plot):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Get data for this stat
    if stat == 'Salary_2025_26':
        data = plot_data[stat] / 1_000_000  # Convert to millions
    else:
        data = plot_data[stat].dropna()
    
    # Create histogram
    n, bins, patches = ax.hist(data, bins=30, alpha=0.7, color=color, edgecolor='black')
    
    # Add mean line
    mean_val = data.mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.2f}')
    
    # Add median line
    median_val = data.median()
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2,
               label=f'Median: {median_val:.2f}')
    
    # Labels and formatting
    ax.set_xlabel(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'{title} Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add stats text box
    stats_text = f'Min: {data.min():.2f}\nMax: {data.max():.2f}\nStd: {data.std():.2f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('stats_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nDistribution plots saved as 'stats_distributions.png'")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS FOR QUALIFIED PLAYERS")
print("="*70)
summary_stats = plot_data[['PER', 'WS', 'VORP', 'BPM', 'TS%']].describe()
print(summary_stats)

print("\n" + "="*70)
print("SALARY DISTRIBUTION (in millions)")
print("="*70)
salary_summary = (plot_data['Salary_2025_26'] / 1_000_000).describe()
print(salary_summary)

# %%
# Cell 31: Box Plots for Advanced Stats Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot 1: Advanced Stats
ax1 = axes[0]
stats_for_box = ['PER', 'TS%', 'WS/48', 'BPM', 'VORP']
box_data = [plot_data[stat].dropna() for stat in stats_for_box]

bp1 = ax1.boxplot(box_data, labels=stats_for_box, patch_artist=True,
                   notch=True, showmeans=True)

# Color the boxes
colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum', 'peachpuff']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
ax1.set_title('Advanced Stats - Box Plots', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticklabels(stats_for_box, rotation=45, ha='right')

# Box plot 2: Salary by Position (if we have position data)
ax2 = axes[1]
if 'Pos' in plot_data.columns:
    # Get top 5 most common positions
    top_positions = plot_data['Pos'].value_counts().head(5).index.tolist()
    salary_by_pos = [plot_data[plot_data['Pos'] == pos]['Salary_2025_26'].dropna() / 1_000_000 
                     for pos in top_positions]
    
    bp2 = ax2.boxplot(salary_by_pos, labels=top_positions, patch_artist=True,
                       notch=True, showmeans=True)
    
    # Color the boxes
    for patch in bp2['boxes']:
        patch.set_facecolor('gold')
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Salary ($M)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Position', fontsize=12, fontweight='bold')
    ax2.set_title('Salary Distribution by Position', fontsize=13, fontweight='bold')
else:
    # Just show overall salary distribution
    bp2 = ax2.boxplot([plot_data['Salary_2025_26'].dropna() / 1_000_000], 
                       labels=['All Players'], patch_artist=True,
                       notch=True, showmeans=True)
    bp2['boxes'][0].set_facecolor('gold')
    bp2['boxes'][0].set_alpha(0.7)
    
    ax2.set_ylabel('Salary ($M)', fontsize=12, fontweight='bold')
    ax2.set_title('Overall Salary Distribution', fontsize=13, fontweight='bold')

ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('boxplots_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nBox plots saved as 'boxplots_comparison.png'")
# %%

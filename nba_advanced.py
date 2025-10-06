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
    
    print(f"Loaded {len(df_merged)} stat rows")
    print(f"Qualified players (15+ MPG): {len(df_merged_qualified)}")
    print("\nYou can now skip to Cell 20 to analyze the data!")
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
table = soup.find('table', {'id': 'advanced_stats'})
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
print(df[['Player', 'Tm', 'PER', 'TS%', 'WS', 'BPM']].describe())

# %%
# Cell 10: Top 10 by PER (Basketball Reference's official PER)
print("\nTop 10 Players by PER:")
top_per = df.nlargest(10, 'PER')[['Player', 'Tm', 'PER', 'TS%', 'WS', 'BPM']]
print(top_per)

# %%
# Cell 11: Top 10 by Win Shares
print("\nTop 10 Players by Win Shares:")
top_ws = df.nlargest(10, 'WS')[['Player', 'Tm', 'MP', 'PER', 'WS', 'WS/48']]
print(top_ws)

# %%
# Cell 12: Filter for qualified players (at least 15 MPG)
df_qualified = df[df['MP'] >= 15].copy()

print(f"\nQualified players (15+ MPG): {len(df_qualified)}")
print(f"Average PER for qualified: {df_qualified['PER'].mean():.2f}")

print("\nTop 10 Qualified Players by PER:")
top_per_qual = df_qualified.nlargest(10, 'PER')[['Player', 'Tm', 'MP', 'PER', 'TS%', 'WS', 'BPM']]
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

# Clean the salary column
df_salary_clean['Salary_2025_26'] = df_salary_clean['2025-26'].str.replace('
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
df_for_ranking['is_tot'] = df_for_ranking['Tm'] == 'TOT'

# Remove duplicate players, keeping TOT if available
df_for_ranking = df_for_ranking.sort_values('is_tot', ascending=False).drop_duplicates(subset='Player', keep='first')

print(f"\nUnique qualified players for ranking: {len(df_for_ranking)}")

# %%
# Cell 21: Top performers by various advanced stats
print("\n" + "="*70)
print("TOP 10 PLAYERS BY PER (with salary)")
print("="*70)
top_per_salary = df_for_ranking.nlargest(10, 'PER')[['Player', 'Tm', 'MP', 'PER', 'TS%', 'WS', 'Salary_2025_26']]
print(top_per_salary)

print("\n" + "="*70)
print("TOP 10 PLAYERS BY WIN SHARES")
print("="*70)
top_ws = df_for_ranking.nlargest(10, 'WS')[['Player', 'Tm', 'MP', 'PER', 'WS', 'WS/48', 'Salary_2025_26']]
print(top_ws)

print("\n" + "="*70)
print("TOP 10 PLAYERS BY BPM (Box Plus/Minus)")
print("="*70)
top_bpm = df_for_ranking.nlargest(10, 'BPM')[['Player', 'Tm', 'MP', 'PER', 'BPM', 'VORP', 'Salary_2025_26']]
print(top_bpm)

# %%
# Cell 22: Value analysis - PER per million
df_for_ranking['Value_PER'] = df_for_ranking['PER'] / (df_for_ranking['Salary_2025_26'] / 1_000_000)
df_for_ranking['Value_PER'] = df_for_ranking['Value_PER'].round(2)

# Also calculate WS per million
df_for_ranking['Value_WS'] = df_for_ranking['WS'] / (df_for_ranking['Salary_2025_26'] / 1_000_000)
df_for_ranking['Value_WS'] = df_for_ranking['Value_WS'].round(2)

print("\n" + "="*70)
print("TOP 10 MOST EFFICIENT SCORERS (True Shooting %)")
print("="*70)
# Filter for players with meaningful scoring (at least 10 PPG in basic stats would be ideal, but we'll use qualified)
top_ts = df_for_ranking.nlargest(10, 'TS%')[['Player', 'Tm', 'TS%', 'PER', 'Salary_2025_26']]
print(top_ts)

# %%
# Cell 24: Save merged data
df_merged.to_csv('nba_advanced_stats_with_salary_2025.csv', index=False)
print("\n" + "="*70)
print("Full data with salary saved to nba_advanced_stats_with_salary_2025.csv")
print("="*70)

# Filter for players making at least $5M
high_earners = df_for_ranking[df_for_ranking['Salary_2025_26'] >= 5_000_000].copy()

print("\n" + "="*70)
print("TOP 10 BEST VALUE PLAYERS - PER per $1M (min $5M salary)")
print("="*70)
top_value_per = high_earners.nlargest(10, 'Value_PER')[['Player', 'Tm', 'PER', 'Salary_2025_26', 'Value_PER']]
print(top_value_per)

print("\n" + "="*70)
print("TOP 10 BEST VALUE PLAYERS - Win Shares per $1M (min $5M salary)")
print("="*70)
top_value_ws = high_earners.nlargest(10, 'Value_WS')[['Player', 'Tm', 'WS', 'Salary_2025_26', 'Value_WS']]
print(top_value_ws)

print("\n" + "="*70)
print("MOST OVERPAID PLAYERS (min $20M salary, lowest PER)")
print("="*70)
overpaid = df_for_ranking[df_for_ranking['Salary_2025_26'] >= 20_000_000].nsmallest(10, 'PER')[['Player', 'Tm', 'PER', 'WS', 'Salary_2025_26', 'Value_PER']]
print(overpaid)

# Cell 23: Efficiency rankings - True Shooting %
print("\n" + "="*70)
print("TOP 10 MOST EFFICIENT SCORERS (True Shooting %)")
print("="*70)
# Filter for players with meaningful scoring (at least 10 PPG in basic stats would be ideal, but we'll use qualified)
top_ts = df_for_ranking.nlargest(10, 'TS%')[['Player', 'Tm', 'TS%', 'PER', 'Salary_2025_26']]
print(top_ts)

# Cell 24: Save merged data
df_merged.to_csv('nba_advanced_stats_with_salary_2025.csv', index=False)
print("\n" + "="*70)
print("Full data with salary saved to nba_advanced_stats_with_salary_2025.csv")
print("="*70)
, '').str.replace(',', '')
df_salary_clean['Salary_2025_26'] = pd.to_numeric(df_salary_clean['Salary_2025_26'], errors='coerce')

# Drop the raw column
df_salary_clean = df_salary_clean[['Player', 'Salary_2025_26']].copy()

print(f"\nSalary data cleaned.")
print(f"Total players with salary: {len(df_salary_clean)}")

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

# Cell 20: Handle multi-team players - prefer TOT rows
df_for_ranking = df_merged_qualified.copy()

# Create a preference: TOT rows first, then single team rows
df_for_ranking['is_tot'] = df_for_ranking['Tm'] == 'TOT'

# Remove duplicate players, keeping TOT if available
df_for_ranking = df_for_ranking.sort_values('is_tot', ascending=False).drop_duplicates(subset='Player', keep='first')

print(f"\nUnique qualified players for ranking: {len(df_for_ranking)}")

# Cell 21: Top performers by various advanced stats
print("\n" + "="*70)
print("TOP 10 PLAYERS BY PER (with salary)")
print("="*70)
top_per_salary = df_for_ranking.nlargest(10, 'PER')[['Player', 'Tm', 'MP', 'PER', 'TS%', 'WS', 'Salary_2025_26']]
print(top_per_salary)

print("\n" + "="*70)
print("TOP 10 PLAYERS BY WIN SHARES")
print("="*70)
top_ws = df_for_ranking.nlargest(10, 'WS')[['Player', 'Tm', 'MP', 'PER', 'WS', 'WS/48', 'Salary_2025_26']]
print(top_ws)

print("\n" + "="*70)
print("TOP 10 PLAYERS BY BPM (Box Plus/Minus)")
print("="*70)
top_bpm = df_for_ranking.nlargest(10, 'BPM')[['Player', 'Tm', 'MP', 'PER', 'BPM', 'VORP', 'Salary_2025_26']]
print(top_bpm)

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
top_value_per = high_earners.nlargest(10, 'Value_PER')[['Player', 'Tm', 'PER', 'Salary_2025_26', 'Value_PER']]
print(top_value_per)

print("\n" + "="*70)
print("TOP 10 BEST VALUE PLAYERS - Win Shares per $1M (min $5M salary)")
print("="*70)
top_value_ws = high_earners.nlargest(10, 'Value_WS')[['Player', 'Tm', 'WS', 'Salary_2025_26', 'Value_WS']]
print(top_value_ws)

print("\n" + "="*70)
print("MOST OVERPAID PLAYERS (min $20M salary, lowest PER)")
print("="*70)
overpaid = df_for_ranking[df_for_ranking['Salary_2025_26'] >= 20_000_000].nsmallest(10, 'PER')[['Player', 'Tm', 'PER', 'WS', 'Salary_2025_26', 'Value_PER']]
print(overpaid)

# Cell 23: Efficiency rankings - True Shooting %
print("\n" + "="*70)
print("TOP 10 MOST EFFICIENT SCORERS (True Shooting %)")
print("="*70)
# Filter for players with meaningful scoring (at least 10 PPG in basic stats would be ideal, but we'll use qualified)
top_ts = df_for_ranking.nlargest(10, 'TS%')[['Player', 'Tm', 'TS%', 'PER', 'Salary_2025_26']]
print(top_ts)

# Cell 24: Save merged data
df_merged.to_csv('nba_advanced_stats_with_salary_2025.csv', index=False)
print("\n" + "="*70)
print("Full data with salary saved to nba_advanced_stats_with_salary_2025.csv")
print("="*70)
# NBA Advanced Player Stats and Salary Scraper

#1: Import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

#2: Set up the URL and headers for advanced stats
url = 'https://www.basketball-reference.com/leagues/NBA_2025_advanced.html'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

#3: Make the request for advanced stats
print(f"Fetching data from {url}...")
response = requests.get(url, headers=headers)
print(f"Status code: {response.status_code}")

#4: Parse the HTML
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


#5: Extract column headers
headers_list = []
for th in table.find('thead').find_all('th'):
    headers_list.append(th.text.strip())
print(f"Columns: {headers_list}")

#6: Extract all rows
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

#7: Create DataFrame
df = pd.DataFrame(rows, columns=headers_list)
print(df.head())

#8: Convert numeric columns
# Advanced stats are mostly numeric after the first few columns
numeric_cols = df.columns[5:]  # Stats start after Pos column
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nData types after conversion:")
print(df.dtypes)

#9: Filter for qualified players (at least 15 MPG)
df_qualified = df[df['MP'] >= 15].copy()

print(f"\nQualified players (15+ MPG): {len(df_qualified)}")
print(f"Average PER for qualified: {df_qualified['PER'].mean():.2f}")

print("\nTop 10 Qualified Players by PER:")
top_per_qual = df_qualified.nlargest(10, 'PER')[['Player', 'Team', 'MP', 'PER', 'TS%', 'WS', 'BPM']]
print(top_per_qual)

#10: Save advanced stats to CSV
df.to_csv('nba_advanced_stats_2025.csv', index=False)
print("\nAdvanced stats saved to nba_advanced_stats_2025.csv")

#11: Scrape player salaries
salary_url = 'https://www.basketball-reference.com/contracts/players.html'

print(f"\nFetching salary data from {salary_url}...")
salary_response = requests.get(salary_url, headers=headers)
print(f"Status code: {salary_response.status_code}")

#12: Parse salary table
salary_soup = BeautifulSoup(salary_response.content, 'lxml')
salary_table = salary_soup.find('table', {'id': 'player-contracts'})
print("Salary table found!" if salary_table else "Salary table not found")

#13: Extract salary data with correct column names
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

#14: Create salary DataFrame
df_salary = pd.DataFrame(salary_rows, columns=correct_headers)
print(f"\nSalary columns: {df_salary.columns.tolist()}")
print(f"Shape: {df_salary.shape}")

#15: Clean salary data
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


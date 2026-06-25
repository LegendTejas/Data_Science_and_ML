# Extract a specific table
## For example, page 1 contains Table 1, Table 2. 
## Camelot indexes them starting at 0

import camelot

# Extract all tables from page 1
tables = camelot.read_pdf(
    "sample_tables.pdf",
    pages="1",
    flavor="lattice"
)

print("Tables found:", tables.n)

# Extract the second table (Table 2)
df = tables[1].df

print(df)
import camelot

tables = camelot.read_pdf(
    "sample_tables.pdf",
    pages="all",
    flavor="lattice"
)

print("Total tables:", tables.n)

for i, table in enumerate(tables):
    print(f"\n========== Table {i+1} ==========")
    print(table.df)
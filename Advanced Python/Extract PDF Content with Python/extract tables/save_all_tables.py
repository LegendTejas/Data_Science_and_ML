import camelot

tables = camelot.read_pdf(
    "sample_tables.pdf",
    pages="all",
    flavor="lattice"
)

for i, table in enumerate(tables):
    df = table.df

    # Set the first row as the header
    df.columns = df.iloc[0]

    # Remove the first row (it is now the header)
    df = df.iloc[1:].reset_index(drop=True)

    df.to_csv(f"table_{i+1}.csv", index=False)
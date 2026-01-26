# Creates a file and writes text into it.
# If the file already exists, it will be overwritten.

try:
    with open("sample.txt", "w") as f:
        f.write("First line\n")
        f.write("Second line\n")

    print("File written successfully.")

except IOError as e:
    print("File write error:", e)
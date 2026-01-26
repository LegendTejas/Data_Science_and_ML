# Reads the complete contents of a file.

try:
    with open("sample.txt", "r") as f:
        data = f.read()
        print(data)

except FileNotFoundError:
    print("File not found.")
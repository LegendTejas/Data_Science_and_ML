# Reads a file one line at a time using iteration.

try:
    with open("sample.txt", "r") as f:
        for line in f:
            print(line.strip())

except FileNotFoundError:
    print("File missing.")
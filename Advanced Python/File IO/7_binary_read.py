# Reads raw bytes from a binary file.

try:
    with open("image.bin", "rb") as f:
        data = f.read()
        print(data)

except FileNotFoundError:
    print("Binary file not found.")
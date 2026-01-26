# Renames an existing file.

import os

try:
    os.rename("sample.txt", "renamed.txt")
    print("File renamed.")

except FileNotFoundError:
    print("Original file not found.")
# Deletes a file from disk.

import os

try:
    os.remove("renamed.txt")
    print("File deleted.")

except FileNotFoundError:
    print("File does not exist.")
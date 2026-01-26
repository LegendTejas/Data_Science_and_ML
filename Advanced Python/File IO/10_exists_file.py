# Checks whether a file is present.

import os

if os.path.exists("sample.txt"):
    print("File exists.")
else:
    print("File not found.")
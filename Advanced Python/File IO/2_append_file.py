# Appends new content at the end of an existing file.
# If file already exists, it will not overwrite it
try:
    with open("sample.txt", "a") as f:
        f.write("Appended line\n")

    print("Data appended.")

except IOError as e:
    print("Append failed:", e)
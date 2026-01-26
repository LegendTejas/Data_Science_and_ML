# Writes raw binary data to a file.

try:
    with open("image.bin", "wb") as f:
        f.write(b"\x48\x65\x6C\x6C\x6F")

    print("Binary file created.")

except IOError as e:
    print("Binary write error:", e)
import re
from pdfminer.high_level import extract_pages, extract_text

text = extract_text("sample.pdf")
print(text) 

# Matches one or more letters followed by a single comma and a single space (e.g., "Hello, ")
pattern = re.compile(r"[a-zA-Z]+,{1}\s{1}")
# Regex breakdown:
# [a-zA-Z]+ : one or more alphabetic characters
# ,{1}      : exactly one comma
# \s{1}     : exactly one whitespace character

matches = pattern.findall(text)
names = [n[:-2] for n in matches]
print(names)
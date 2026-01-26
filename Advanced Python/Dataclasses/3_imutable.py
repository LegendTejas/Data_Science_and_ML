from dataclasses import dataclass

# Immutable dataclass (fields can't be changed after creation)
@dataclass(frozen=True)
class BlogPost:
    user: str

b1 = BlogPost('Tejas')
b1.user = "Paul"   # Error: can't modify frozen dataclass
print(b1)
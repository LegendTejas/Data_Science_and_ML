from dataclasses import dataclass, field   # Import dataclass and field for customization

@dataclass
class BlogPost:
    user: str
    content: str                           # Field for post content
    # content: str = field(repr=False)     # Hides 'content' from __repr__ output if uncommented
    likes: int = 0                         # Field with default value 0

b1 = BlogPost('Tejas', 'This is my post. Welcome Guys!')  # Create BlogPost instance
print(b1)                                                 # Print instance (auto-formatted by dataclass)
from dataclasses import dataclass, field

# kw_only=True -> forces all fields to be specified as keyword arguments during instantiation
@dataclass(frozen=True, kw_only=True)
class BlogPost:
    user: str                              # Required field
    content: str = field(repr=False)       # Field hidden from __repr__ output
    expiry: int = 24                       # Default value in hours
    likes: int = 0                          # Default value

# Must use keyword arguments because kw_only=True
b1 = BlogPost(user='Tejas', content='This is my post. Welcome Guys!', expiry=48, likes=0)

print(b1)  # Output: BlogPost(user='Tejas', expiry=48, likes=0)
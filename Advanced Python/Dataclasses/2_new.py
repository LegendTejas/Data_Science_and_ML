from dataclasses import dataclass

@dataclass
class BlogPost:
    user: str

b1 = BlogPost('Tejas')
print(b1)
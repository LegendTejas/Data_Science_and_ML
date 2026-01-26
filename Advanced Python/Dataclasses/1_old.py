class BlogPost:
    def __init__(self, user: str):
        self.user = user

    def __repr__(self):
        return f"BlogPost(user={self.user})"

b1 = BlogPost('Tejas')
print(b1)
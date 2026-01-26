from dataclasses import dataclass


@dataclass(frozen=True)
class BlogPost:
    user: str
    likes: int = 0

b1 = BlogPost('Tejas')
print(b1)


# # without dataclass we had to define default values like this which is a bit hectic:
# class BlogPost:
#     def __init__(self, user: str, likes = 0):
#         self.user = user
#         self.likes = likes

#     def __repr__(self):
#         return f"BlogPost(user={self.user})"

# b1 = BlogPost('Tejas')
# print(b1)
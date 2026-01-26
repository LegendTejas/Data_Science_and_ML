from dataclasses import dataclass

@dataclass(order=True)
class Point:
    x: int
    y: int

print(Point(1, 2) < Point(2, 1))  # True
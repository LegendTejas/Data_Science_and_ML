# Dataclasses in Python

In Python, dataclasses (introduced in Python 3.7) are a decorator-based way to simplify class creation when your class is mainly used to store data — similar to a record or struct.


**They automatically generate common boilerplate code like:**

* `__init__()`

* `__repr__()`

* `__eq__()`

* `__hash__()` (if applicable)

### Why Dataclasses Simplify Code

| Without Dataclass                                              | With Dataclass                                                    |
| -------------------------------------------------------------- | ----------------------------------------------------------------- |
| You must manually write `__init__`, `__repr__`, `__eq__`, etc. | Automatically generated                                           |
| Risk of typos and repetitive code                              | Cleaner, declarative, and less error-prone                        |
| Harder to maintain large data models                           | Easy to read and maintain                                         |
| Limited built-in immutability or ordering                      | Supports immutability (`frozen=True`) and ordering (`order=True`) |

---

### Basic Example — Without and With Dataclass

1. Without `@dataclass`

Here, you manually define `__init__` and `__repr__`

```
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def __repr__(self):
        return f"Student(name={self.name}, age={self.age}, grade={self.grade})"

# Usage
s1 = Student("Tejas", 21, "A")
print(s1)
```

2. With `@dataclass`

```
from dataclasses import dataclass

@dataclass
class Student:
    name: str
    age: int
    grade: str

# Usage
s1 = Student("Tejas", 21, "A")
print(s1)
```
The dataclass automatically provides `__init__` , `__repr__` , and `__eq__`


#### More Examples:

You can check `old.py` (without dataclass) and `new.py` (with dataclass)

---

### Features of Datclasses

1. Saves a lot of time

2. Allows Imutability((use frozen=True) Example: check the `imutable.py` file)

3. Default values are easier (Example: check `default.py`)

4. Full control of attribute types (Example: check `attr.py`)

5. Comparison Support(Example: check `compare.py`)

6. Force Keywords [Python3.10 and above] (Example: check `force.py`)


---

### Limitations of dataclasses:

1. Not ideal for classes with complex logic or behavior — they’re best for data storage.

2. Mutable by default, which can cause issues in hashing unless `frozen=True`.

3. Inheritance can get tricky, especially with default and non-default fields.

4. Don’t support custom `__init__` logic easily without losing auto-generated features.

5. Not as memory-efficient as `__slots__`-based classes or lightweight alternatives like namedtuple.

---

### When to use dataclasses:

- ✅ When your class mainly stores and represents data (like models, records, or configurations).


- ❌ Avoid when the class has complex behavior, logic, or requires fine control over initialization or immutability.
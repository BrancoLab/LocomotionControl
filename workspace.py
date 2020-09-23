# %%
from rich.syntax import Syntax
from rich import print

with open("proj/model/config.py") as f:
    content = f.read()

content

# %%
print(Syntax(content, "python", theme="monokai", line_numbers=True))

# %%

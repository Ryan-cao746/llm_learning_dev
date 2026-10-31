import os
from pathlib import Path
import tiktoken

"""tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

strings = tokenizer.decode(integers)

print(strings)
"""

def construct():

    text = get_raw_text()

    tokenizer = tiktoken.get_encoding("gpt2")

    enc_text = tokenizer.encode(text)
    print(len(enc_text))

def get_project_root() -> Path:
    """从当前文件向上查找包含项目入口文件的目录。"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "main.py").exists():
            return parent
    raise FileNotFoundError("未找到包含 main.py 的项目目录")

def get_raw_text():
    project_root = get_project_root()
    dataset_root = project_root / "dataset"

    text = ""

    for text_file in dataset_root.rglob("*.txt"):
        text += text_file.read_text(encoding="utf-8")

    return text

if __name__ == "__main__":
    construct()
    
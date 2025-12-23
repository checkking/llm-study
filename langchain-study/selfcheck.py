
import importlib

packages = ["langchain", "langchain_core", "langchain_community"]
for pkg in packages:
    try:
        m = importlib.import_module(f"{pkg}.output_parsers")
        if hasattr(m, "ResponseSchema"):
            print(f"找到啦！请使用: from {pkg}.output_parsers import ResponseSchema")
            break
    except ImportError:
        continue
else:
    print("在常用路径中未找到。请尝试: from langchain.output_parsers.structured import ResponseSchema")
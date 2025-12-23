import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
# 这个包在所有 0.1+ 版本中都非常稳定
from langchain_core.output_parsers import JsonOutputParser

# 1. 初始化模型
model = init_chat_model(
    model="doubao-1-5-pro-32k-250115",
    model_provider="openai",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_OPENAI_API_KEY"),
)

# 2. 初始化 JsonOutputParser (不需要定义复杂的 ResponseSchema)
parser = JsonOutputParser()

# 3. 编写链
# 第一步生成文本
gen_prompt = PromptTemplate.from_template("请根据菜名编写一个简短的制作步骤, 包括制作难度与耗时：{dish_name}")
# 第二步提取并强制要求 JSON
extract_prompt = PromptTemplate.from_template(
    "从以下内容中提取信息：食材清单(ingredients)、烹饪时长(time)、难度(difficulty)。\n"
    "必须以 JSON 格式返回。\n"
    "内容：{recipe}"
)

# 4. 组合链
full_chain = (
    {"recipe": gen_prompt | model | (lambda x: x.content)}
    | extract_prompt
    | model
    | parser  # 直接解析 JSON
)

# 执行
try:
    result = full_chain.invoke({"dish_name": "地三鲜"})
    print(result)
except Exception as e:
    print(f"仍然报错: {e}")
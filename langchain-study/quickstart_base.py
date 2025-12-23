# 1. 导入必要的模块
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import os


# 2. 将函数定义为 LangChain 的 Tool
def get_weather(city: str) -> str:
    """Get the current weather in a given city."""
    # 这里是模拟，真实应用中应接入天气API
    return f"The weather in {city} is sunny and 72°F."

# 3. 以 OpenAI 兼容模式初始化聊天模型
# 关键：通过指定 base_url 来兼容不同服务商
# 1) Deepseek 模型
llm = ChatOpenAI(
    model="deepseek-chat",  # 替换为豆包的模型名，如 "Doubao-lite-32k"
    openai_api_key=os.environ["OPENAI_API_KEY"],  # 替换为你的API密钥
    base_url="https://api.deepseek.com/v1",  # 替换为豆包的API端点
    temperature=0
)
# 2) 火山引擎Doubao模型
llm_doubao = ChatOpenAI(
    model="doubao-1-5-pro-32k-250115",  # 替换为豆包的模型名，如 "Doubao-lite-32k"
    openai_api_key=os.environ["ARK_OPENAI_API_KEY"],  # 替换为你的API密钥
    base_url="https://ark.cn-beijing.volces.com/api/v3",  # 替换为火山引擎的API端点
    temperature=0
)

# 4. 定义智能体的系统提示词
system_prompt = "You are a helpful assistant that can provide weather information."

# 5. 创建智能体
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt=system_prompt
)

# 6. 执行查询
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
)

print(result)

# 7 创建doubao智能体
agent_doubao = create_agent(
    model=llm_doubao,
    tools=[get_weather],
    system_prompt=system_prompt
)

# 8. 执行查询
result_doubao = agent_doubao.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
)
print(result_doubao)

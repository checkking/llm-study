import requests
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model



@tool
def get_weather(loc: str):
    """查询即时天气函数，根据输入的城市名称，查询对应城市的实时天气"""
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": os.environ["XINZHI_API_KEY"],
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data["results"][0]["now"]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是天气助手，请根据用户的问题，给出相应的天气信息"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Doubao / 火山模型
model = init_chat_model(
    model="doubao-1-5-pro-32k-250115",
    model_provider="openai",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ["ARK_OPENAI_API_KEY"],
)

# 创建 agent
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是天气助手，请根据用户的问题，给出相应的天气信息",
)


# 调用
result = agent.invoke(
    {"messages": [("user", "北京的天气")]}
)


final_message = result["messages"][-1]
print(final_message.content)

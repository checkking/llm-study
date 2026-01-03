import requests
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model


class WeatherQuery(BaseModel):
    loc: str = Field(description="城市名称")


@tool(args_schema=WeatherQuery)
def get_weather(loc):
    """
    查询即时天气函数
    :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，
    :return：心知天气, 返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": os.environ.get("XINZHI_API_KEY"),
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature['results'][0]['now']


def main():
    """主函数：创建并运行天气助手智能体"""
    # 初始化大模型
    model = init_chat_model(
        model="doubao-1-5-pro-32k-250115",
        model_provider="openai",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=os.environ.get("ARK_OPENAI_API_KEY"),
    )

    # 创建ReACT预制图结构并构建智能体
    agent = create_react_agent(
        model=model,
        tools=[get_weather]
    )

    # 调用智能体
    res = agent.invoke(
        {
            "messages": [
                ("user", "你好，请介绍一下你自己，并告诉我北京的天气。")
            ]
        }
    )

    return res


if __name__ == "__main__":
    result = main()
    print("智能体输出结果：")
    print(result)
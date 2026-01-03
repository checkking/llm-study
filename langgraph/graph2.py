import os
import requests
from datetime import datetime
from typing import Dict

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model


# ========= 工具 1：天气查询 =========
class WeatherQuery(BaseModel):
    loc: str = Field(description="城市名称")


@tool(args_schema=WeatherQuery)
def get_weather(loc: str) -> Dict:
    """查询城市实时天气"""
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": os.environ.get("XINZHI_API_KEY"),
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    print(params)
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    print(data)
    return data["results"][0]["now"]


# ========= 工具 2：当前时间 =========
@tool
def get_time() -> str:
    """获取当前系统时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ========= 工具 3：城市基础信息（示例工具） =========
class CityInfoQuery(BaseModel):
    city: str = Field(description="城市名称")


@tool(args_schema=CityInfoQuery)
def get_city_info(city: str) -> Dict:
    """获取城市的基础介绍信息（示例）"""
    mock_db = {
        "北京": {
            "country": "中国",
            "role": "首都",
            "population": "约 2180 万",
            "feature": "政治、文化、国际交往中心",
        },
        "上海": {
            "country": "中国",
            "role": "经济、国际交往中心",
            "population": "约 2480 万",
            "feature": "经济、国际交往中心",
        },
    }
    return mock_db.get(city, {"info": "暂无该城市信息"})


def main():
    # ========= 初始化大模型 =========
    model = init_chat_model(
        model="doubao-1-5-pro-32k-250115",
        model_provider="openai",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=os.environ.get("ARK_OPENAI_API_KEY"),
        temperature=0.2,
    )

    # ========= 创建 LangGraph ReAct 智能体 =========
    agent = create_react_agent(
        model=model,
        tools=[
            get_weather,
            get_time,
            get_city_info,
        ],
    )

    # ========= 触发多工具 + 自主规划 =========
    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "你好，请分别告诉我北京、上海现在的天气，当前时间，以及城市的基本情况。"
                )
            ]
        }
    )
    print(result)
    print("====== 智能体最终输出 ======")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

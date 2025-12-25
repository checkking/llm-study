import os
import requests
import json
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser # 导入标准输出组件

model = init_chat_model(
    model="doubao-1-5-pro-32k-250115",
    model_provider="openai",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ["ARK_OPENAI_API_KEY"], #你注册的火山引擎api_key
)

# 查询天气
def get_weather(loc):
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": os.environ["XINZHI_API_KEY"], #填写你的私钥
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature['results'][0]['now']


get_weather_function = {
    'name': 'get_weather',
    'description': '查询即时天气函数，根据输入的城市名称，查询对应城市的实时天气',
    'parameters': {
        'type': 'object',
        'properties': { #参数说明
            'loc': {
                'description': '城市名称',
                'type': 'string'
            }
        },
        'required': ['loc']  #必备参数
    }
}


tools = [
    {
        "type": "function",
        "function":get_weather_function
    }
]

available_functions = {
    "get_weather": get_weather,
}


# 搭建链条，把model和字符串输出解析器组件连接在一起
basic_qa_chain = model
# 查看输出结果
messages=[
    {"role": "user", "content": "请帮我查询北京地区今日天气情况"}
]
result = basic_qa_chain.invoke(messages, tools=tools, tool_choice="auto")

print(result)

# 获取函数名称
function_name = result.tool_calls[0]["name"]

# 获得对应函数对象
function_to_call = available_functions[function_name]

# 获得执行函数所需参数
function_args = result.tool_calls[0]["args"]

# 执行函数
function_response = function_to_call(**function_args)

print(function_response)
messages.append(result.model_dump()) 
messages.append({
    "role": "tool",
    "content": json.dumps(function_response), # 将回复的字典转化为json字符串
    "tool_call_id": result.tool_calls[0]["id"] # 将函数执行结果作为tool_message添加到messages中, 并关联返回执行函数内容的id
})

second_response = model.invoke(messages)
print(second_response.content)





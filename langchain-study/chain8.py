import os
import requests
import json
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

# 初始化模型（使用豆包模型）
model = init_chat_model(
    model="doubao-1-5-pro-32k-250115",
    model_provider="openai",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ["ARK_OPENAI_API_KEY"],
)

# 查询天气函数
def get_weather(loc):
    """查询即时天气函数，根据输入的城市名称，查询对应城市的实时天气"""
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": os.environ["XINZHI_API_KEY"],  # 填写你的私钥
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature['results'][0]['now']

# 另一个示例函数：获取城市信息
def get_city_info(city):
    """获取城市基本信息"""
    # 这里可以接入其他API，比如获取城市人口、面积等信息
    city_data = {
        "北京": {"population": "2154万", "area": "16410平方公里", "description": "中华人民共和国的首都"},
        "上海": {"population": "2489万", "area": "6341平方公里", "description": "中国的经济中心城市"},
        "广州": {"population": "1530万", "area": "7434平方公里", "description": "广东省省会，重要的港口城市"},
    }
    return city_data.get(city, {"error": "未找到该城市信息"})

def run_conv(messages, 
             api_key, 
             tools=None, 
             functions_list=None, 
             model_name="doubao-1-5-pro-32k-250115",
             max_tool_calls=5):
    """
    通用的对话函数，支持工具调用（增强版，支持多次调用）
    
    Args:
        messages: 对话消息列表
        api_key: API密钥
        tools: 工具列表（可选）
        functions_list: 可用函数列表（可选）
        model_name: 模型名称
        max_tool_calls: 最大工具调用次数，防止无限循环
    
    Returns:
        str: 最终响应内容
    """
    # 设置环境变量以确保API key正确
    import os
    original_api_key = os.environ.get("ARK_OPENAI_API_KEY")
    os.environ["ARK_OPENAI_API_KEY"] = api_key
    
    try:
        # 使用现有的全局模型实例
        user_messages = messages.copy()
        
        # 如果没有外部函数库，则执行普通的对话任务
        if tools is None:
            response = model.invoke(user_messages)
            final_response = response.content
        
        # 若存在外部函数库，支持多次工具调用
        else:
            # 创建外部函数库字典
            available_functions = {func.__name__: func for func in functions_list}
            
            # 循环处理工具调用
            tool_call_count = 0
            
            while tool_call_count < max_tool_calls:
                # 调用模型
                response = model.invoke(
                    user_messages, 
                    tools=tools, 
                    tool_choice="auto"
                )
                response_message = response
                
                # 检查是否有工具调用
                if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                    # 处理所有工具调用
                    for tool_call in response_message.tool_calls:
                        # 获取函数名和参数
                        function_name = tool_call["name"]
                        function_args = tool_call["args"]
                        tool_call_id = tool_call["id"]
                        
                        # 检查函数是否存在
                        if function_name in available_functions:
                            function_to_call = available_functions[function_name]
                            
                            # 执行函数
                            try:
                                function_response = function_to_call(**function_args)
                                
                                # 将模型响应和工具响应添加到消息历史
                                user_messages.append(response_message.model_dump())
                                user_messages.append({
                                    "role": "tool",
                                    "content": json.dumps(function_response, ensure_ascii=False),
                                    "tool_call_id": tool_call_id
                                })
                                
                            except Exception as e:
                                print(f"函数执行出错: {e}")
                                # 添加错误信息到消息历史
                                user_messages.append(response_message.model_dump())
                                user_messages.append({
                                    "role": "tool",
                                    "content": json.dumps({"error": str(e)}),
                                    "tool_call_id": tool_call_id
                                })
                        else:
                            print(f"未找到函数: {function_name}")
                    
                    tool_call_count += 1
                    
                else:
                    # 模型未选择工具调用，结束循环
                    break
            
            # 获取最终响应
            if user_messages:
                final_response = response_message.content
            else:
                final_response = "未能获得有效响应"
        
        return final_response
    
    finally:
        # 恢复原来的API key
        if original_api_key is not None:
            os.environ["ARK_OPENAI_API_KEY"] = original_api_key

# 定义工具
get_weather_function = {
    'name': 'get_weather',
    'description': '查询即时天气函数，根据输入的城市名称，查询对应城市的实时天气',
    'parameters': {
        'type': 'object',
        'properties': {  # 参数说明
            'loc': {
                'description': '城市名称',
                'type': 'string'
            }
        },
        'required': ['loc']  # 必备参数
    }
}

get_city_info_function = {
    'name': 'get_city_info',
    'description': '获取城市基本信息，包括人口、面积和描述',
    'parameters': {
        'type': 'object',
        'properties': {
            'city': {
                'description': '城市名称',
                'type': 'string'
            }
        },
        'required': ['city']
    }
}

tools = [
    {
        "type": "function",
        "function": get_weather_function
    },
    {
        "type": "function", 
        "function": get_city_info_function
    }
]

# 测试代码
if __name__ == "__main__":
    # 设置API密钥
    ds_api_key = os.environ["ARK_OPENAI_API_KEY"]
    
    # 测试1：普通对话（无工具）
    print("=== 测试1：普通对话 ===")
    messages1 = [{"role": "user", "content": "请介绍一下人工智能的发展历史"}]
    response1 = run_conv(messages=messages1, api_key=ds_api_key)
    print(f"AI回答：{response1}")
    print()
    
    # 测试2：工具调用（天气查询）
    print("=== 测试2：天气查询 ===")
    messages2 = [{"role": "user", "content": "请问上海今天天气如何？"}]
    response2 = run_conv(messages=messages2, 
                        api_key=ds_api_key,
                        tools=tools, 
                        functions_list=[get_weather, get_city_info])
    print(f"AI回答：{response2}")
    print()
    
    # 测试3：工具调用（城市信息）
    print("=== 测试3：城市信息查询 ===")
    messages3 = [{"role": "user", "content": "请告诉我北京的基本信息"}]
    response3 = run_conv(messages=messages3,
                        api_key=ds_api_key, 
                        tools=tools,
                        functions_list=[get_weather, get_city_info])
    print(f"AI回答：{response3}")
    print()
    
    # 测试4：复合问题（需要多个工具）
    print("=== 测试4：复合问题（多次工具调用） ===")
    messages4 = [{"role": "user", "content": "我想了解广州的天气和基本信息"}]
    print(f"问题: {messages4[0]['content']}")
    print("开始处理...")
    
    response4 = run_conv(messages=messages4,
                        api_key=ds_api_key,
                        tools=tools, 
                        functions_list=[get_weather, get_city_info],
                        max_tool_calls=5)
    
    print(f"AI回答：{response4}")
    print("=" * 50)
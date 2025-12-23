import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

# 使用 硅基流动 模型
model = init_chat_model(
    model="doubao-1-5-pro-32k-250115",
    model_provider="openai",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_OPENAI_API_KEY"),
)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你叫小智，是人工智能专家。"),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model | parser

messages_list = []  # 初始化历史
print("🔹 输入 exit 结束对话")
while True:
    user_query = input("你：")
    if user_query.lower() in {"exit", "quit"}:
        break

    # 1) 追加用户消息
    messages_list.append(HumanMessage(content=user_query))

    # 2) 调用模型
    assistant_reply = ''
    print("小智：", end=' ')
    for chunk in chain.stream({"messages": messages_list}):
        assistant_reply+=chunk
        print(chunk, end="", flush=True)
    print()

    # 3) 追加 AI 回复
    messages_list.append(AIMessage(content=assistant_reply))

    # 4) 仅保留最近 50 条
    messages_list = messages_list[-50:]

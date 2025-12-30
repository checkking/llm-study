import os
import asyncio
import csv
import json
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import init_chat_model

# CSV 写入逻辑
def save_to_csv(data: dict, filename="news_data.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['提炼标题', '总结内容', 'URL'])
        writer.writerow([data.get('title'), data.get('summary'), data.get('url')])
    return f"成功保存到 {filename}"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools = toolkit.get_tools()

        # --- 调试：打印所有工具名，防止名称不匹配 ---
        tool_names = [t.name for t in tools]
        print(f"当前可用工具: {tool_names}")

        # 动态查找工具，增加容错
        def find_tool(names):
            for name in names:
                tool = next((t for t in tools if t.name == name), None)
                if tool: return tool
            return None

        navigate_tool = find_tool(["navigate_browser", "navigate"])
        # 这里适配可能的不同命名：extract_text 或 get_text
        extract_tool = find_tool(["extract_text", "get_text"])

        if not navigate_tool or not extract_tool:
            raise RuntimeError(f"未能找到核心浏览器工具。当前工具列表: {tool_names}")

        # --- 初始化模型 ---
        model = init_chat_model(
            model="doubao-1-5-pro-32k-250115",
            model_provider="openai",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.environ["ARK_OPENAI_API_KEY"],
        )

        prompt = ChatPromptTemplate.from_template("""
        你是一个新闻提取专家。以下是从网页抓取的原始文本内容：
        ---
        {page_content}
        ---
        请提取该网页的新闻标题和核心内容摘要。
        必须以 JSON 格式输出，包含以下键：
        {{
            "title": "提炼的标题",
            "summary": "100字以内的总结内容",
            "url": "{url}"
        }}
        不要输出任何 Markdown 代码块包裹，只输出纯 JSON。
        """)

        chain = prompt | model | JsonOutputParser()

        try:
            target_url = "https://baijiahao.baidu.com/s?id=1852806076428347030"
            print(f"正在启动浏览器访问: {target_url}")

            # 串行执行
            await navigate_tool.ainvoke({"url": target_url})
            # 增加一点等待时间确保页面渲染（可选）
            await asyncio.sleep(2) 
            
            raw_text = await extract_tool.ainvoke({})
            print(f"成功抓取文本，长度: {len(raw_text)} 字符")
            
            # AI 处理
            result = await chain.ainvoke({
                "page_content": raw_text[:10000], 
                "url": target_url
            })

            # 保存
            save_to_csv(result)
            print("CSV 导出成功！")

        except Exception as e:
            print(f"运行过程中出错: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
    
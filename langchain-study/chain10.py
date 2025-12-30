import os
import asyncio
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

async def main():
    # 1. 直接使用 Playwright 官方 API 启动，避开 LangChain 内部的 loop 冲突
    async with async_playwright() as p:
        # 启动浏览器 (headless=True 表示无头模式，False 可以看到浏览器界面)
        browser = await p.chromium.launch(headless=True)
        
        try:
            # 2. 将原生的 playwright browser 传入工具包
            toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
            tools = toolkit.get_tools()
            print(f"成功加载 {len(tools)} 个浏览器工具")

            # 模型配置
            model = init_chat_model(
                model="doubao-1-5-pro-32k-250115",
                model_provider="openai",
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=os.environ["ARK_OPENAI_API_KEY"],
            )

            system_message = """你是一个有用的AI助手，可以使用浏览器工具来访问网页、提取内容等。
            请根据用户的问题，合理使用工具来完成任务。"""

            # 注意：新版 LangChain 推荐使用 create_tool_calling_agent 或类似的封装
            # 这里沿用你代码中的 create_agent 逻辑
            agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=system_message,
            )

            print("=== PlayWright Browser Agent 测试 ===")
            
            # 3. 运行 Agent
            result = await agent.ainvoke(
                {"messages": [("user", "请访问 https://baijiahao.baidu.com/s?id=1852806076428347030，提取并总结其内容")]}
            )
            
            print(result["messages"][-1].content)

        except Exception as e:
            print(f"运行出错: {e}")
        finally:
            # 4. 关闭浏览器
            await browser.close()

if __name__ == "__main__":
    # 在标准 Python 脚本中运行
    asyncio.run(main())
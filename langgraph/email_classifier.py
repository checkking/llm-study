import os
import json
from typing import TypedDict, Literal, Annotated, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages

# --- 模拟外部依赖和异常 ---
class SearchAPIError(Exception): pass
class ToolError(Exception): pass

def fetch_customer_history(customer_id: str):
    return {"customer_id": customer_id, "tier": "premium", "last_purchase": "2025-12-01"}

def run_tool(tool_name: str):
    return f"Result from {tool_name}"

# --- 定义状态结构 ---
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str
    customer_id: Optional[str]
    classification: Optional[EmailClassification]
    search_results: Optional[List[str]]
    customer_history: Optional[Dict[str, Any]]
    draft_response: Optional[str]
    # 使用 Annotated[..., add_messages] 让消息列表支持追加模式
    messages: Annotated[List[BaseMessage], add_messages]

# --- 节点函数定义 ---

def read_email(state: EmailAgentState):
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "draft_response", "bug_tracking", "lookup_customer_history"]]:
    """分类意图"""
    
    # 强制 JSON 模式
    structured_llm = llm.with_structured_output(EmailClassification, method="json_mode")
    
    classification_prompt = f"""你是一个邮件分类助手。请直接输出 JSON 格式。
    必须包含: intent (question, bug, billing, feature, complex), urgency (low, medium, high, critical), topic, summary.
    
    邮件内容: "{state['email_content']}"
    """

    try:
        classification = structured_llm.invoke(classification_prompt)
    except Exception:
        classification = {"intent": "complex", "urgency": "high", "topic": "error", "summary": "parse_error"}

    # 打印分类结果，方便调试
    print(f"\n[AI 分类结果] Intent: {classification['intent']} | Urgency: {classification['urgency']}")

    # 更新状态中的分类信息
    # 注意：这里删除了直接跳转 'human_review' 的逻辑，确保所有邮件都有草稿
    if classification['intent'] == 'billing':
        goto = "lookup_customer_history"
    elif classification['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        # 即使是 Critical，也先去 Draft 生成回复，再在 Draft 节点决定是否需要人工
        goto = "draft_response"

    return Command(
        update={"classification": classification},
        goto=goto
    )

def lookup_customer_history(state: EmailAgentState) -> Command[Literal["draft_response", "lookup_customer_history"]]:
    if not state.get('customer_id'):
        # 触发人工干预获取 ID
        user_input = interrupt({
            "message": "Customer ID needed",
            "request": "Please provide account ID"
        })
        return Command(
            update={"customer_id": user_input['customer_id']},
            goto="lookup_customer_history"
        )
    
    customer_data = fetch_customer_history(state['customer_id'])
    return Command(update={"customer_history": customer_data}, goto="draft_response")

def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    classification = state.get('classification', {})
    # 模拟搜索逻辑
    search_results = ["Reset password via Settings", "Include uppercase and symbols"]
    return Command(update={"search_results": search_results}, goto="draft_response")

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    ticket_id = "BUG-12345"
    return Command(
        update={"search_results": [f"Bug ticket {ticket_id} created"]},
        goto="draft_response"
    )

def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    classification = state.get('classification', {}) or {}
    context = []
    if state.get('search_results'):
        context.append(f"Docs: {state['search_results']}")
    if state.get('customer_history'):
        context.append(f"History: {state['customer_history']}")

    draft_prompt = f"Draft a response to: {state['email_content']}\nContext: {' '.join(context)}"
    response = llm.invoke(draft_prompt)

    needs_review = classification.get('urgency') in ['high', 'critical'] or classification.get('intent') == 'billing'
    goto = "human_review" if needs_review else "send_reply"

    return Command(update={"draft_response": response.content}, goto=goto)

def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """生成草稿，并根据紧急程度决定是否转人工"""
    classification = state.get('classification', {}) or {}
    
    # 构建上下文
    context = []
    if state.get('search_results'): context.append(f"Docs: {state['search_results']}")
    if state.get('customer_history'): context.append(f"History: {state['customer_history']}")
    
    draft_prompt = f"""
    为以下邮件起草回复 (Intent: {classification.get('intent')}, Urgency: {classification.get('urgency')})。
    用户邮件: {state['email_content']}
    参考信息: {context}
    """
    
    response = llm.invoke(draft_prompt)
    print(f"-> 草稿已生成: {response.content[:50]}...")

    # --- 关键路由逻辑 ---
    # 如果紧急度是 High/Critical，或者是复杂的 Billing 问题，强制人工审核
    is_urgent = classification.get('urgency') in ['high', 'critical']
    is_billing = classification.get('intent') == 'billing'
    
    if is_urgent or is_billing:
        print(f"!!! 触发人工干预 (原因: Urgent={is_urgent}, Billing={is_billing}) !!!")
        goto = "human_review"
    else:
        goto = "send_reply"

    return Command(update={"draft_response": response.content}, goto=goto)


def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    # 触发人工审查中断
    human_decision = interrupt({
        "original_email": state.get('email_content'),
        "draft_response": state.get('draft_response'),
        "action": "Approve or Edit"
    })

    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response'))},
            goto="send_reply"
        )
    return Command(goto=END)

def send_reply(state: EmailAgentState):
    print(f"\n--- [EMAIL SENT] ---\nTo: {state['sender_email']}\nContent: {state['draft_response']}\n--------------------\n")
    return {"messages": [AIMessage(content="Email sent successfully.")]}

# --- 图构建 ---

# 设置环境变量 (请确保你已经设置了正确的 API Key)
# os.environ["ARK_OPENAI_API_KEY"] = "your-key-here"

llm = init_chat_model(
    model="doubao-1-5-pro-32k-250115",
    model_provider="openai",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_OPENAI_API_KEY")
)

workflow = StateGraph(EmailAgentState)

workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("lookup_customer_history", lookup_customer_history)
workflow.add_node("search_documentation", search_documentation, retry_policy=RetryPolicy(max_attempts=3))
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

app = workflow.compile(checkpointer=MemorySaver())

# --- 执行示例 ---

def run_test_scenario(scenario_name: str, email_input: dict):
    print(f"\n{'='*20} 测试场景: {scenario_name} {'='*20}")
    
    # 为每个测试用例分配独立的 thread_id
    config = {"configurable": {"thread_id": f"thread_{scenario_name}"}}
    
    # 第一次调用
    result = app.invoke(email_input, config)
    
    # 循环处理所有的中断，直到流程结束
    while True:
        state = app.get_state(config)
        
        # 检查是否有中断挂起
        if not state.next: # 没有下一个节点，说明执行结束
            print(f"[{scenario_name}] 流程顺利结束。")
            break
            
        # 如果当前状态有中断信息
        if "__interrupt__" in state.values:
            interrupt_data = state.values["__interrupt__"][0].value
            print(f"[收到中断请求]: {interrupt_data.get('message') or interrupt_data.get('action')}")
            
            # 自动化模拟人工回复逻辑
            if "customer_id" in str(interrupt_data).lower() or "account ID" in str(interrupt_data):
                resume_value = {"customer_id": "TEST-CUST-666"}
                print(f"-> 自动填入客户ID: {resume_value}")
            else:
                resume_value = {"approved": True, "edited_response": "这是经过人工确认的最终回复。"}
                print(f"-> 自动执行人工审批通过")
            
            # 恢复执行
            app.invoke(Command(resume=resume_value), config)
        else:
            # 如果没有中断但流程没结束，可能是正常的节点间跳转（在 invoke 中会自动完成）
            break


def run_interactive_test():
    print("\n🚀 启动紧急人工干预交互测试...")
    
    input_data = {
        "email_content": "This is a CRITICAL emergency! My production server is down and I'm losing money every second! Help!",
        "sender_email": "vip_client@example.com",
        "email_id": "urgent_test_01"
    }
    
    config = {"configurable": {"thread_id": "interactive_test_thread"}}
    
    # 1. 启动流程
    print("--- 流程开始 ---")
    app.invoke(input_data, config)
    
    # 2. 检查循环
    while True:
        # 获取当前状态快照
        state = app.get_state(config)
        
        # A. 检查流程是否已经彻底结束 (没有下一步了)
        if not state.next:
            print("✅ 流程结束")
            break
            
        # B. 关键修改：从 tasks 中检查是否有中断挂起
        # 在 LangGraph 中，使用 interrupt() 函数产生的中断位于 tasks[0].interrupts 中
        potential_interrupts = state.tasks[0].interrupts if state.tasks else []
        
        if potential_interrupts:
            # 获取第一个中断的内容
            interrupt_info = potential_interrupts[0].value
            
            print(f"\n🛑 [系统暂停] 需要人工介入!")
            print(f"    原因/请求: {interrupt_info.get('action')}")
            print(f"    当前草稿: \n    '{interrupt_info.get('draft_response')}'\n")
            
            # --- 等待用户输入 ---
            user_choice = input("👉 请输入指令 (y: 批准发送 / n: 拒绝 / edit: 修改): ").strip().lower()
            
            if user_choice == 'y':
                resume_payload = {"approved": True}
                print("-> 已批准，继续发送...")
            elif user_choice == 'edit':
                new_text = input("请输入修改后的回复: ")
                resume_payload = {"approved": True, "edited_response": new_text}
                print("-> 修改已提交，继续发送...")
            else:
                resume_payload = {"approved": False}
                print("-> 已拒绝，流程将终止。")
            
            # 使用 resume 恢复执行
            app.invoke(Command(resume=resume_payload), config)
        
        else:
            # 如果没有下一步，也没有中断，这通常是不可能的（除非invoke在没有做任何事的情况下返回）
            print("⚠️ 状态异常：既未结束也无中断，正在退出...")
            break

if __name__ == '__main__':
    # 定义测试用例集
    test_cases = [
        {
            "name": "账单分支测试",
            "input": {
                "email_content": "I see an extra charge on my bill for $29.99.",
                "sender_email": "billing_user@example.com",
                "email_id": "case_001"
            }
        },
        {
            "name": "知识库搜索测试",
            "input": {
                "email_content": "Can you tell me the password requirements?",
                "sender_email": "info_user@example.com",
                "email_id": "case_002"
            }
        },
        {
            "name": "Bug跟踪分支测试",
            "input": {
                "email_content": "The 'Export to PDF' button is not working.",
                "sender_email": "dev_user@example.com",
                "email_id": "case_003"
            }
        },
        {
            "name": "紧急人工干预测试",
            "input": {
                "email_content": "URGENT: My account is locked and I have a deadline in 10 minutes!",
                "sender_email": "urgent_user@example.com",
                "email_id": "case_004"
            }
        }
    ]

    run_interactive_test()

    #for case in test_cases:
    #    run_test_scenario(case["name"], case["input"])
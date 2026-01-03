from typing import Literal, TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

# 1. 定义状态模型
class CustomerServiceState(BaseModel):
    customer_type: Literal["VIP", "regular"]  # 客户类型
    issue_complexity: Literal["simple", "complex"]  # 问题复杂度
    response: str = ""  # 响应内容
    assigned_agent: str = ""  # 分配的客服

# 2. 定义节点函数
def analyze_customer(state: CustomerServiceState) -> CustomerServiceState:
    print(f"[客户分析] 客户类型: {state.customer_type}, 问题复杂度: {state.issue_complexity}")
    return state  # 此节点只做分析，不修改状态

def route_request(state: CustomerServiceState) -> str:
    """条件路由函数，决定下一步流向哪个节点"""
    if state.customer_type == "VIP":
        return "senior_agent"
    elif state.issue_complexity == "simple":
        return "ai_assistant"
    else:
        return "regular_agent"

def senior_agent(state: CustomerServiceState) -> CustomerServiceState:
    print("[高级客服] 处理VIP客户请求")
    return CustomerServiceState(
        customer_type=state.customer_type,
        issue_complexity=state.issue_complexity,
        response="您的请求已由高级客服处理，我们会尽快解决。",
        assigned_agent="senior_agent"
    )

def regular_agent(state: CustomerServiceState) -> CustomerServiceState:
    print("[普通客服] 处理复杂问题")
    return CustomerServiceState(
        customer_type=state.customer_type,
        issue_complexity=state.issue_complexity,
        response="您的问题已分配给专业客服，将在24小时内回复。",
        assigned_agent="regular_agent"
    )

def ai_assistant(state: CustomerServiceState) -> CustomerServiceState:
    print("[AI助手] 自动回复简单问题")
    return CustomerServiceState(
        customer_type=state.customer_type,
        issue_complexity=state.issue_complexity,
        response="根据我们的知识库，您的问题可以通过以下步骤解决：...",
        assigned_agent="ai_assistant"
    )

# 3. 构建图
builder = StateGraph(CustomerServiceState)

# 添加节点
builder.add_node("analyze", analyze_customer)
builder.add_node("senior_agent", senior_agent)
builder.add_node("regular_agent", regular_agent)
builder.add_node("ai_assistant", ai_assistant)

# 设置入口
builder.add_edge(START, "analyze")

# 添加条件边
builder.add_conditional_edges(
    "analyze",
    route_request,  # 路由函数
    {
        "senior_agent": "senior_agent",
        "regular_agent": "regular_agent",
        "ai_assistant": "ai_assistant"
    }
)

# 所有路径都指向结束
builder.add_edge("senior_agent", END)
builder.add_edge("regular_agent", END)
builder.add_edge("ai_assistant", END)

# 4. 编译图
graph = builder.compile()

# 5. 可视化
graph.get_graph().print_ascii()

# 6. 测试不同场景
print("\n===== 测试案例 1: VIP客户 =====")
result1 = graph.invoke(CustomerServiceState(customer_type="VIP", issue_complexity="complex"))
print(f"结果: {result1}")

print("\n===== 测试案例 2: 普通客户 + 简单问题 =====")
result2 = graph.invoke(CustomerServiceState(customer_type="regular", issue_complexity="simple"))
print(f"结果: {result2}")

print("\n===== 测试案例 3: 普通客户 + 复杂问题 =====")
result3 = graph.invoke(CustomerServiceState(customer_type="regular", issue_complexity="complex"))
print(f"结果: {result3}")
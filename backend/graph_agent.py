from typing import TypedDict, List, Literal
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

# --- CONFIGURATION ---
LLM_MODEL = "mistral" # Ensure this model supports tool calling (Mistral v0.3+ is best)

# 1. Define the State of the Agent
class AgentState(TypedDict):
    messages: List[BaseMessage]
    final_verdict: dict

# 2. Define the Agent Logic
def build_agent_graph(tools):
    """Builds a state machine for a specific bidder."""
    
    # Initialize the Local LLM with tools bound to it
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # NODE: The Brain (Decides to search or answer)
    def reasoner(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # NODE: The Tools (Executes the search)
    tool_node = ToolNode(tools)

    # LOGIC: Should we loop back or finish?
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    # Build the Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", reasoner)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent") # Loop back to agent after searching

    return workflow.compile()

# 3. Helper to run the audit
def run_audit(req_id, req_text, bidder_name, tools):
    app = build_agent_graph(tools)
    
    system_prompt = """You are a strict AI Auditor. Your goal is to verify if the bidder meets the requirement.
    1. First, SEARCH the Tender Requirements to ensure you understand the rule.
    2. Second, SEARCH the Bidder Documents for evidence.
    3. If evidence is missing, search for synonyms (e.g., "Balance Sheet" instead of "Financial Statement").
    4. FINAL OUTPUT must be valid JSON with fields: {"status": "Pass/Fail/NA", "reasoning": "...", "quote": "..."}.
    Do not output any text before or after the JSON."""

    user_msg = f"Requirement {req_id}: {req_text}"
    
    # Run the graph
    result = app.invoke({
        "messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_msg)],
        "final_verdict": {}
    })
    
    # Extract final text
    return result["messages"][-1].content

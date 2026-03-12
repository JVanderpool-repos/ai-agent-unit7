import os
import asyncio
import json
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.types import Command

load_dotenv()

# 1. Define shared state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Global variable for the researcher agent (will be set in main)
researcher_agent = None

async def researcher_node(state: State) -> Command[Literal["__end__"]]:
    """Research node that hands off to writer."""
    print("\n" + "="*50)
    print("RESEARCHER NODE")
    print("="*50)
    
    response = await researcher_agent.ainvoke({"messages": state["messages"]})
    
    # Debug: Print search results and tool usage
    print("\n--- Research Results ---")
    for msg in response["messages"]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"\nTool Called: {tool_call.get('name', 'Unknown')}")
                print(f"Arguments: {tool_call.get('args', {})}")
        
        if getattr(msg, 'type', None) == "tool":
            print(f"\nTool Response from: {getattr(msg, 'name', 'Unknown Tool')}")
            content_preview = str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content)
            print(f"Content: {content_preview}")
        
        if getattr(msg, 'type', None) == "ai" and not getattr(msg, 'tool_calls', None):
            print(f"\nResearcher Response:")
            print(f"{msg.content}")
    
    print("\n" + "="*50 + "\n")
    
    return Command(
        update={"messages": response["messages"]},
        goto="__end__"
    )

async def main():
    """Test researcher agent only."""
    global researcher_agent
    
    # Check API keys
    if not os.getenv("GITHUB_TOKEN"):
        print("Error: GITHUB_TOKEN not found in .env")
        return
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found in .env")
        return
    
    # 1. Initialize LLM (AFTER API checks)
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.7,
        base_url="https://models.github.ai/inference",
        api_key=os.getenv("GITHUB_TOKEN")
    )
    
    # 2. Load researcher template
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(script_dir, "templates", "researcher.json")
        with open(template_path, "r") as f:
            researcher_data = json.load(f)
            researcher_prompt = researcher_data.get("template", "You are a helpful research assistant.")
        print("✅ Loaded researcher template")
    except FileNotFoundError:
        print(f"❌ Template missing: {template_path}")
        return
    
    # 3. Setup MCP/Tavily
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    research_client = MultiServerMCPClient({
        "tavily": {
            "transport": "http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}",
        }
    })
    
    researcher_tools = await research_client.get_tools()
    print(f"✅ Research tools: {[tool.name for tool in researcher_tools]}")
    
    # 4. Create researcher agent
    researcher_agent = create_agent(
        llm, 
        tools=researcher_tools, 
        system_prompt=researcher_prompt
    )
    print("✅ Researcher agent ready")
    
    # 5. Build + test graph
    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node)
    builder.add_edge(START, "researcher")
    graph = builder.compile()
    
    print("\n" + "="*50)
    print("🧪 TESTING RESEARCHER")
    print("="*50)
    
    user_input = input("Enter research topic: ")
    initial_message = HumanMessage(content=user_input)
    result = await graph.ainvoke({"messages": [initial_message]})
    
    print("\n✅ Researcher test complete!")
    print("Final summary:", result["messages"][-1].content[:200] + "...")

if __name__ == "__main__":
    asyncio.run(main())

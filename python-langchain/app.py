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
writer_agent = None
editor_agent = None

async def researcher_node(state: State) -> Command[Literal["writer", "__end__"]]:
    """Research node that hands off to writer."""
    print("\n" + "="*50)
    print("RESEARCHER NODE")
    print("="*50)
    trimmed_messages = state["messages"][-3:]
    response = await researcher_agent.ainvoke({"messages": trimmed_messages})

    
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
        goto="writer"
    )

async def writer_node(state: State) -> Command[Literal["editor", "__end__"]]:
    """Writer node that hands off to editor."""
    print("\n" + "="*50)
    print("WRITER NODE")
    print("="*50)
    
    response = await writer_agent.ainvoke({"messages": state["messages"]})
    
    # Print the written content
    final_message = response["messages"][-1]
    print(f"\nWriter Output:")
    print(f"{final_message.content}")
    print("\n" + "="*50 + "\n")
    
    # Native handoff: explicitly tell the graph to move to 'editor'
    return Command(
        update={"messages": response["messages"]},
        goto="editor"
    )

async def editor_node(state: State) -> Command[Literal["writer", "__end__"]]:
    """Editor node that can hand back to writer or end."""
    print("\n" + "="*50)
    print("EDITOR NODE")
    print("="*50)
    
    response = await editor_agent.ainvoke({"messages": state["messages"]})
    
    # Debug: Print editor feedback
    final_message = response["messages"][-1]
    print(f"\nEditor Feedback:")
    print(f"{final_message.content}")
    
    # Example logic: if editor finds an error, hand back to writer
    if "REVISE" in str(final_message.content):
        print("\n⚠️  Editor requested REVISION - routing back to writer")
        print("="*50 + "\n")
        return Command(
            update={"messages": response["messages"]},
            goto="writer"
        )
    
    print("\n✓ Editor approved - workflow complete")
    print("="*50 + "\n")
    
    return Command(
        update={"messages": response["messages"]},
        goto="__end__"
    )    
async def main():
    """Run the multi-agent content creation workflow."""
    global researcher_agent, writer_agent, editor_agent
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
        api_key=os.getenv("GITHUB_TOKEN"),
        max_tokens=1000
    )
    
    # 2. Load researcher template
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(script_dir, "templates", "researcher.json")
        with open(template_path, "r") as f:
            researcher_data = json.load(f)
            researcher_prompt = researcher_data.get("template", "You are a helpful research assistant.")
        print("✅ Loaded researcher template")

        # ADD THESE:
        with open(os.path.join(script_dir, "templates", "writer.json"), "r") as f:
            writer_prompt = json.load(f).get("template", "You are a helpful writing assistant.")
        print("✅ Loaded writer template")

        with open(os.path.join(script_dir, "templates", "editor.json"), "r") as f:
            editor_prompt = json.load(f).get("template", "You are a helpful editing assistant.")
        print("✅ Loaded editor template")

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
    writer_agent = create_agent(
        llm, 
        tools=[],
        system_prompt=writer_prompt
    )

    editor_agent = create_agent(
        llm, 
        tools=[], 
        system_prompt=editor_prompt
    )
    print("✅ Writer and Editor agents ready")
    # 5. Build + test graph
        # Build the Graph without manual edges (Edgeless Handoff)
    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("editor", editor_node)
    
    # Only need to set the entry point
    builder.add_edge(START, "researcher")
    graph = builder.compile()
    
    print("\n" + "="*50)
    print("Starting Multi-Agent Content Creation Workflow")
    print("="*50)
    
    user_input = input("Enter research topic: ")
    initial_message = HumanMessage(content=user_input)
    result = await graph.ainvoke({"messages": [initial_message]})
    
    print("\n" + "="*50)
    print("✅ Workflow Complete!")
    print("="*50 + "\n")
    print("Final Output:")
    print(result["messages"][-1].content if result["messages"] else "No output")

if __name__ == "__main__":
    asyncio.run(main())

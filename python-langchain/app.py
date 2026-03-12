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
    revision_count: int
# Global variable for the researcher agent (will be set in main)
researcher_agent = None
writer_agent = None
fact_checker_agent = None
editor_agent = None

async def researcher_node(state: State) -> Command[Literal["writer", "__end__"]]:
    """Research node that hands off to writer."""
    print("\n" + "="*50)
    print("RESEARCHER NODE")
    print("="*50)
    trimmed_messages = state["messages"][-1:]
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

async def writer_node(state: State) -> Command[Literal["fact_checker", "__end__"]]:
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
        goto="fact_checker"   
    )
async def fact_checker_node(state: State) -> Command[Literal["editor", "__end__"]]:
    """Fact-checker node that reviews writer content before editor."""
    print("\n" + "="*50)
    print("FACT-CHECKER NODE")
    print("="*50)

    response = await fact_checker_agent.ainvoke({"messages": state["messages"]})

    final_message = response["messages"][-1]
    print(f"\nFact-Checker Output:")
    print(f"{final_message.content}")
    print("\n" + "="*50 + "\n")

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

    final_message = response["messages"][-1]
    print(f"\nEditor Feedback:")
    print(f"{final_message.content}")

    # Get current revision count
    revision_count = state["revision_count"]

    if "REVISE" in str(final_message.content) and revision_count < 2:
        print(f"\n⚠️  Revision requested ({revision_count + 1}/2) - routing back to writer")
        print("="*50 + "\n")
        return Command(
            update={
                "messages": response["messages"],
                "revision_count": revision_count + 1
            },
            goto="writer"
        )

    if revision_count >= 2:
        print("\n⚠️  Max revisions (2) reached - forcing approval")
    
    article_content = None
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
            article_content = msg.content
            break

    if article_content:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "article.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(article_content)
        print(f"\n📄 Article saved to: {output_path}")
        
    print("\n✓ Editor approved - workflow complete")
    print("="*50 + "\n")
    return Command(
        update={"messages": response["messages"]},
        goto="__end__"
    )

async def main():
    """Run the multi-agent content creation workflow."""
    global researcher_agent, writer_agent, fact_checker_agent, editor_agent  # ← all here
    
    # Check API keys
    if not os.getenv("GITHUB_TOKEN"):
        print("Error: GITHUB_TOKEN not found in .env")
        return
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found in .env")
        return
    
    # 1. Initialize LLM
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.7,
        base_url="https://models.github.ai/inference",
        api_key=os.getenv("GITHUB_TOKEN"),
        max_tokens=1000
    )
    
    # 2. Load all templates
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(script_dir, "templates", "researcher.json"), "r") as f:
            researcher_prompt = json.load(f).get("template", "You are a helpful research assistant.")
        print("✅ Loaded researcher template")
        
        with open(os.path.join(script_dir, "templates", "writer.json"), "r") as f:
            writer_prompt = json.load(f).get("template", "You are a helpful writing assistant.")
        print("✅ Loaded writer template")
        
        with open(os.path.join(script_dir, "templates", "fact_checker.json"), "r") as f:
            fact_checker_prompt = json.load(f).get("template", "You are a fact-checking assistant.")
        print("✅ Loaded fact-checker template")
        
        with open(os.path.join(script_dir, "templates", "editor.json"), "r") as f:
            editor_prompt = json.load(f).get("template", "You are a helpful editing assistant.")
        print("✅ Loaded editor template")

    except FileNotFoundError as e:
        print(f"❌ Template missing: {e}")
        return


    
    # 3. Setup MCP/Tavily
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    research_client = MultiServerMCPClient({ #type: ignore
        "tavily": {
            "transport": "http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}",
        },
        "wikipedia": {
            "transport": "stdio",
            "command": "wikipedia-mcp",
            "args": ["--transport", "stdio"],
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
    fact_checker_agent = create_agent(
        llm,
        tools=[],
        system_prompt=fact_checker_prompt
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
    builder.add_node("fact_checker", fact_checker_node)
    builder.add_node("editor", editor_node)
    
    # Only need to set the entry point
    builder.add_edge(START, "researcher")
    graph = builder.compile()
    
    print("\n" + "="*50)
    print("Starting Multi-Agent Content Creation Workflow")
    print("="*50)
    
    user_input = input("Enter research topic: ")
    initial_message = HumanMessage(content=user_input)
    async for chunk in graph.astream(
        {"messages": [initial_message], "revision_count": 0},
        stream_mode="values"
    ):
        final_state = chunk  # each chunk is the full state after a node fires

    print("\n" + "="*50)
    print("✅ Workflow Complete!")
    print("="*50 + "\n")
    print("Final Output:")
    if final_state and final_state.get("messages"):
        print(final_state["messages"][-1].content)
    else:
        print("No output")

if __name__ == "__main__":
    asyncio.run(main())

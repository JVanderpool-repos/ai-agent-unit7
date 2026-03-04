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


# TODO: Add node functions for researcher, writer, and editor
# (You'll add these in the lab)

# Global variable for the researcher agent (will be set in main)
researcher_agent = None


async def researcher_node(state: State) -> Command[Literal["writer", "__end__"]]:
    """Research node that hands off to writer."""
    print("\n" + "="*50)
    print("RESEARCHER NODE")
    print("="*50)
    
    # Get the researcher agent from the closure
    response = await researcher_agent.ainvoke({"messages": state["messages"]})
    
    # Debug: Print search results and tool usage
    print("\n--- Research Results ---")
    for msg in response["messages"]:
        # Check for tool calls (AI messages with tool_calls)
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"\nTool Called: {tool_call.get('name', 'Unknown')}")
                print(f"Arguments: {tool_call.get('args', {})}")
        
        # Check for tool responses (ToolMessage)
        if msg.type == "tool":
            print(f"\nTool Response from: {getattr(msg, 'name', 'Unknown Tool')}")
            content_preview = str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content)
            print(f"Content: {content_preview}")
        
        # Print AI responses (but not tool calls)
        if msg.type == "ai" and not hasattr(msg, 'tool_calls'):
            print(f"\nResearcher Response:")
            print(f"{msg.content}")
    
    print("\n" + "="*50 + "\n")
    
    # Native handoff: explicitly tell the graph to move to 'writer'
    return Command(
        update={"messages": response["messages"]},
        goto="writer"
    )

async def main():
    """Run the multi-agent content creation workflow."""
    
    # Check for required API keys
    if not os.getenv("GITHUB_TOKEN"):
        print("Error: GITHUB_TOKEN not found.")
        print("Add GITHUB_TOKEN=your-token to a .env file")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found.")
        print("Add TAVILY_API_KEY=your-key to a .env file")
        print("Get your API key from: https://app.tavily.com/")
        return
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.7,
        base_url="https://models.github.ai/inference",
        api_key=os.getenv("GITHUB_TOKEN")
    )
    
    # TODO: Load templates, create MCP client, create agents, and build graph
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(script_dir, "templates", "researcher.json")
        
        with open(template_path, "r") as f:
            researcher_data = json.load(f)
            # This pulls the 'template' string from your JSON file
            researcher_prompt = researcher_data.get("template", "You are a helpful research assistant.")
            print("Successfully loaded researcher template.")
    except FileNotFoundError:
        print(f"Error: Template file not found at {template_path}")
        return
    
    # Get Tavily API key from environment
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    # Create MCP client for Tavily
    research_client = MultiServerMCPClient({
        "tavily": {
            "transport": "http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}",
        }
    })
    
    # Get tools from the client (await because it's async)
    researcher_tools = await research_client.get_tools()
    
    print(f"Research tools: {[tool.name for tool in researcher_tools]}")
    
    # Create agents using create_agent (new API)
    global researcher_agent
    researcher_agent = create_agent(
        llm, 
        tools=researcher_tools, 
        system_prompt=researcher_prompt
    )   
    
    print("\nSetup complete! Follow the lab instructions to build your multi-agent workflow.")


if __name__ == "__main__":
    asyncio.run(main())

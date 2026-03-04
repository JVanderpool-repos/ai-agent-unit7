# Lab: Building a Multi-Agent Content Creation Workflow

## Overview

In this lab, you'll build a multi-agent AI system that collaborates to research, write, and edit content. You'll learn how to:
- Use **LangGraph** to orchestrate multiple AI agents
- Implement **edgeless handoffs** between agents using the Command pattern
- Integrate **MCP (Model Context Protocol)** for web search capabilities
- Create specialized agents with custom prompts loaded from templates

By the end of this lab, you'll have a working system where:
1. A **Researcher** agent searches the web for information
2. A **Writer** agent creates content based on the research
3. An **Editor** agent reviews and either approves or sends back for revision

## Prerequisites

### Required API Keys

You'll need two API keys for this lab:

1. **GitHub Token** - for accessing GitHub Models (GitHub AI)
   - Go to https://github.com/settings/tokens
   - Create a personal access token with appropriate permissions
   
2. **Tavily API Key** - for web search capabilities
   - Sign up at https://tavily.com
   - Get your API key from https://app.tavily.com/

### Environment Setup

Create a `.env` file in the `python-langchain` directory with:

```env
GITHUB_TOKEN=your-github-token-here
TAVILY_API_KEY=your-tavily-api-key-here
```

### Install Dependencies

Make sure you have all required packages installed:

```bash
pip install -r requirements.txt
```

---

## Part 1: Setup the Orchestration

In this first part, we'll set up the foundation of our multi-agent system: the graph structure that will orchestrate our agents.

### Step 1.1: Import Required Libraries

Open `app.py` and start with the following imports:

```python
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
```

**What's happening here?**
- `TypedDict` and `Annotated` help define our state structure with type hints
- `StateGraph` is the core LangGraph component for building agent workflows
- `Command` enables "edgeless handoffs" - agents can explicitly route to other agents
- `add_messages` is a reducer function that properly combines messages in our state

### Step 1.2: Define the Shared State

All agents in our workflow will share a common state. Add this code:

```python
# 1. Define shared state
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**Understanding the State:**
- `messages`: A list of all conversation messages (user input, agent responses, tool calls, etc.)
- `Annotated[list, add_messages]`: The `add_messages` reducer ensures messages are properly merged when nodes update the state
- This shared state allows agents to see the full conversation history and build upon each other's work

### Step 1.3: Create the Main Function Structure

Now let's create the main function that will orchestrate everything:

```python
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
    
    # We'll add more here in the next steps
    
    print("\nOrchestration setup complete!")


if __name__ == "__main__":
    asyncio.run(main())
```

**What's happening here?**
- We validate that API keys are present before proceeding
- We create a ChatOpenAI instance pointing to GitHub Models
- `temperature=0.7` provides a good balance of creativity and consistency
- The `asyncio.run(main())` pattern allows us to use async/await for handling the MCP client

### Step 1.4: Understanding the Workflow Architecture

Before we build the agents, let's understand how the workflow will operate:

```
User Input
    ↓
Researcher (searches web, gathers information)
    ↓
Writer (creates content based on research)
    ↓
Editor (reviews content)
    ↓
    ├─→ Approved? → END
    └─→ Needs revision? → Back to Writer
```

**Key Concepts:**

1. **Edgeless Handoffs**: Instead of defining all edges upfront, each node returns a `Command` that specifies where to go next
2. **State Accumulation**: Each agent adds to the shared message history
3. **Dynamic Routing**: The editor can decide whether to approve or request revisions

**Test Your Understanding:**
- Run `python app.py` now. You should see "Orchestration setup complete!"
- If you see any errors, check your imports and API keys

---

## Part 2: Add the Researcher Agent

Now we'll create our first agent: the Researcher. This agent will use Tavily's web search to gather information.

### Step 2.1: Load the Researcher Template

Agent prompts are stored in JSON template files. Let's load the researcher template. Add this code to your `main()` function, after initializing the LLM:

```python
    # Load prompts from your local filesystem
    with open("templates/researcher.json", "r") as f:
        researcher_data = json.load(f)
        researcher_prompt = researcher_data.get("template", "You are a helpful research assistant.")
```

**What's in the template?**
Open `templates/researcher.json` to see the prompt. It instructs the agent to:
- Use search tools to find accurate, current information
- Gather comprehensive facts and data
- Present findings clearly for the writer to use

### Step 2.2: Setup the MCP Client for Tavily

MCP (Model Context Protocol) provides a standard way to give agents access to external tools. Add this code:

```python
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
```

**Understanding MCP:**
- MCP provides a standardized protocol for tools (like web search)
- `MultiServerMCPClient` can connect to multiple MCP servers
- Tavily exposes its search API through MCP
- `await research_client.get_tools()` retrieves the available tools (like `tavily_search`)

### Step 2.3: Create the Researcher Agent

Now create the agent with the LLM, tools, and prompt:

```python
    # Create agents using create_agent (new API)
    researcher_agent = create_agent(
        llm, 
        tools=researcher_tools, 
        system_prompt=researcher_prompt
    )
```

**What's happening:**
- `create_agent()` is LangChain's new API for creating agentic executors
- The agent can use the provided tools (Tavily search) to answer questions
- The system prompt guides the agent's behavior and objectives

### Step 2.4: Create the Researcher Node Function

Nodes are the individual steps in your workflow. Add this function BEFORE the `main()` function:

```python
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
```

**Understanding the Node Function:**

1. **Input**: Receives the current `State` (which includes all messages)
2. **Processing**: 
   - Invokes the researcher agent with the message history
   - The agent will use Tavily search tools as needed
   - Prints debug information about tool usage
3. **Output**: Returns a `Command` object:
   - `update`: Adds the new messages to the state
   - `goto`: Explicitly routes to the "writer" node next

**The Command Pattern (Edgeless Handoff):**
- Traditional workflows require predefined edges: `graph.add_edge("researcher", "writer")`
- With `Command`, each node decides where to go next at runtime
- Type hint `Command[Literal["writer", "__end__"]]` shows possible destinations

### Step 2.5: Build the Graph and Test the Researcher

Now let's build the graph and test just the researcher. Update your `main()` function to set the global variable and build the graph:

```python
async def main():
    """Run the multi-agent content creation workflow."""
    global researcher_agent  # Add this line at the top of main()
    
    # ... (previous code for API keys, LLM, templates, MCP client) ...
    
    # Create agents using create_agent (new API)
    researcher_agent = create_agent(
        llm, 
        tools=researcher_tools, 
        system_prompt=researcher_prompt
    )
    
    # Build the Graph
    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node)
    
    # Set the entry point
    builder.add_edge(START, "researcher")
    graph = builder.compile()
    
    # Run the workflow
    print("\n" + "="*50)
    print("Starting Multi-Agent Content Creation Workflow")
    print("="*50 + "\n")
    
    user_input = input("Enter the topic that you would like to research: ")
    initial_message = HumanMessage(content=user_input)
    result = await graph.ainvoke({"messages": [initial_message]})
    
    print("\n" + "="*50)
    print("Workflow Complete")
    print("="*50 + "\n")
    print("Final Output:")
    print(result["messages"][-1].content if result["messages"] else "No output")
```

**Understanding the Graph:**
- `StateGraph(State)` creates a graph that manages our State type
- `add_node("researcher", researcher_node)` adds our researcher node
- `add_edge(START, "researcher")` sets the entry point
- `graph.compile()` builds the executable workflow

**Test It:**
1. Run `python app.py`
2. Enter a topic like "latest developments in quantum computing"
3. Watch the researcher:
   - Call the Tavily search tool
   - Gather information
   - Present research findings

You'll see it try to handoff to "writer" but that node doesn't exist yet - that's expected!

---

## Part 3: Add Writer and Editor Agents

Now let's complete the workflow by adding the writer and editor agents.

### Step 3.1: Load Writer and Editor Templates

Add this code in `main()` after loading the researcher template:

```python
    with open("templates/writer.json", "r") as f:
        writer_data = json.load(f)
        writer_prompt = writer_data.get("template", "You are a helpful writing assistant.")
    
    with open("templates/editor.json", "r") as f:
        editor_data = json.load(f)
        editor_prompt = editor_data.get("template", "You are a helpful editing assistant.")
```

**Explore the Templates:**
- Open `templates/writer.json` - Note how it instructs the agent to create well-structured content
- Open `templates/editor.json` - Note how it looks for quality issues and can request revisions

### Step 3.2: Create Writer and Editor Agents

Add this code after creating the researcher agent:

```python
    # Writer and editor don't need tools
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
```

**Note:** Writer and editor don't need external tools - they work with the information already in the message history.

Also, update the global variables at the top of the file:

```python
# Global variables for agents (will be set in main)
researcher_agent = None
writer_agent = None
editor_agent = None
```

And in `main()`, update the global declaration:

```python
async def main():
    """Run the multi-agent content creation workflow."""
    global researcher_agent, writer_agent, editor_agent
    # ... rest of the code
```

### Step 3.3: Create the Writer Node

Add this function before `main()`:

```python
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
```

**Understanding the Writer:**
- Takes all previous messages (including research findings)
- Creates content based on the research
- Always hands off to the editor for review
- Simpler than researcher - no tool calls needed

### Step 3.4: Create the Editor Node with Conditional Logic

Add this function before `main()`:

```python
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
```

**Understanding the Editor's Logic:**

This is where the workflow becomes dynamic:

1. **Review**: The editor examines the writer's content
2. **Decision**: 
   - If the content contains "REVISE" → routes back to writer for improvements
   - Otherwise → approves and ends the workflow with `"__end__"`
3. **Loop Prevention**: The editor prompt should guide when to approve vs. request revision

**Important Note:** The editor template should instruct the agent to include "REVISE" in its response when requesting changes.

### Step 3.5: Complete the Graph with All Nodes

Update the graph building section in `main()`:

```python
    # Build the Graph without manual edges (Edgeless Handoff)
    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("editor", editor_node)
    
    # Only need to set the entry point
    builder.add_edge(START, "researcher")
    graph = builder.compile()
```

**Notice:**
- We only define ONE edge: `START → researcher`
- All other routing is handled by `Command` returns in the node functions
- This is "edgeless handoff" - the workflow is more flexible and self-directing

### Step 3.6: Test the Complete Workflow

Run your complete application:

```bash
python app.py
```

**Try these test inputs:**
1. "Recent advancements in renewable energy"
2. "The impact of AI on healthcare"
3. "Latest space exploration missions"

**Watch the workflow:**
1. Researcher searches the web and gathers information
2. Writer creates an article based on the research
3. Editor reviews and either:
   - Approves (workflow ends)
   - Requests revision (goes back to writer)

---

## Understanding What You Built

### The Complete Architecture

```
┌─────────────────────────────────────────────┐
│           User Input (Question)              │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
        ┌──────────────────┐
        │   RESEARCHER     │
        │  - Tavily Search │
        │  - Gather Info   │
        └────────┬─────────┘
                 │ Command(goto="writer")
                 ▼
        ┌──────────────────┐
        │     WRITER       │
        │  - Create Draft  │
        │  - Format Content│
        └────────┬─────────┘
                 │ Command(goto="editor")
                 ▼
        ┌──────────────────┐
        │     EDITOR       │
        │  - Review Draft  │
        │  - Check Quality │
        └────────┬─────────┘
                 │
         ┌───────┴────────┐
         │                │
    "REVISE"           Approved
    found?
         │                │
         ▼                ▼
   Command(goto=     Command(goto=
    "writer")         "__end__")
```

### Key Concepts Mastered

1. **State Management**
   - Shared state across all agents
   - Message accumulation with `add_messages` reducer
   - Type-safe state definition with `TypedDict`

2. **Agent Creation**
   - Using `create_agent()` to combine LLM + tools + prompts
   - Loading prompts from JSON templates
   - Specialized agents for different tasks

3. **MCP Integration**
   - Connecting to external tools via Model Context Protocol
   - Using Tavily for web search capabilities
   - Tool invocation and response handling

4. **Edgeless Handoffs**
   - Using `Command` for dynamic routing
   - Node functions that decide next steps
   - Flexible workflows without predefined edges

5. **Async Patterns**
   - Using `async`/`await` for agent invocations
   - Handling MCP client asynchronously
   - Running the graph with `ainvoke()`

---

## Extension Challenges

Want to take this further? Try these enhancements:

### Challenge 1: Add a Fact-Checker Agent
Create a fourth agent that reviews the writer's content for factual accuracy before the editor sees it.

**Hint:** Add a `fact_checker_node` between writer and editor.

### Challenge 2: Implement Revision Limits
Prevent infinite loops by limiting revisions to a maximum of 2.

**Hint:** Add a `revision_count` field to the State.

### Challenge 3: Add Multiple Research Sources
Extend the researcher to use multiple MCP servers (e.g., add a Wikipedia MCP server).

**Hint:** Look at the `MultiServerMCPClient` configuration.

### Challenge 4: Stream Output
Instead of waiting for the complete response, stream each agent's output as it's generated.

**Hint:** Use `graph.astream()` instead of `ainvoke()`.

### Challenge 5: Save the Final Article
Write the approved content to a file with proper formatting.

**Hint:** In the editor node, when approved, save `final_message.content` to a file.

---

## Troubleshooting

### Common Issues

**Issue: "TAVILY_API_KEY not found"**
- Solution: Make sure your `.env` file is in the `python-langchain` directory
- Verify the key name is exactly `TAVILY_API_KEY`

**Issue: Agent doesn't use search tools**
- Check that `researcher_tools` is not empty
- Verify your Tavily API key is valid
- Look at the researcher template - it should encourage tool use

**Issue: Infinite loop between writer and editor**
- Check the editor template - it should provide clear approval criteria
- Verify the "REVISE" keyword detection logic
- Consider adding a revision counter

**Issue: "Command" errors**
- Make sure you return `Command` objects from node functions
- Verify the `goto` parameter matches an existing node name or `"__end__"`
- Check type hints match possible destinations

---

## Summary

Congratulations! You've built a sophisticated multi-agent AI system that:

✅ Orchestrates multiple specialized agents with LangGraph  
✅ Integrates external tools via MCP (Tavily search)  
✅ Implements dynamic routing with edgeless handoffs  
✅ Manages shared state across agents  
✅ Uses template-based prompting for consistent behavior  
✅ Handles async operations properly  

This pattern can be extended to many other multi-agent workflows: coding assistants, customer service bots, data analysis pipelines, and more.

---

## Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Tavily API Documentation](https://docs.tavily.com/)
- [LangChain Agents Guide](https://python.langchain.com/docs/modules/agents/)

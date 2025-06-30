# Week 7 Lecture: Building Agentic LLM Applications

## Overview

Welcome to Week 7! This week we'll explore the exciting world of **Agentic LLM Applications** - where LLMs go beyond simple text generation to become intelligent agents that can reason, plan, and take actions using tools and external resources.

## Learning Objectives

By the end of this lecture, you will be able to:

* Understand what makes an LLM "agentic" and how it differs from traditional LLM usage
* Learn about the core components of agentic systems: reasoning, planning, and tool use
* Explore the ReAct framework and how it enables agents to think and act
* Understand how to build agents using LangChain and other frameworks
* Create simple agents that can use tools like calculators, web search, and custom functions

## 1. What is an Agentic LLM?

### Traditional LLMs vs Agentic LLMs

**Traditional LLM Usage:**
- Single-turn interactions
- Generate text based on input
- No memory of previous interactions
- No ability to use external tools or resources
- Limited to pre-trained knowledge

**Agentic LLMs:**
- Multi-turn reasoning and planning
- Can use tools and external resources
- Maintain context and memory across interactions
- Can break down complex tasks into steps
- Can take actions and observe results

### The Agentic Paradigm

An agentic LLM follows this core loop:

```
Thought → Action → Observation → Thought → Action → ...
```

1. **Thought**: The agent reasons about what to do next
2. **Action**: The agent takes an action (e.g., use a tool, make a decision)
3. **Observation**: The agent observes the result of the action
4. **Repeat**: The agent continues this cycle until the task is complete

## 2. Core Components of Agentic Systems

### 2.1 Reasoning Engine (The "Brain")

The LLM serves as the reasoning engine that:
- Analyzes the current situation
- Plans the next steps
- Decides which tools to use
- Interprets results from actions

### 2.2 Tool Interface (The "Hands")

Tools allow the agent to:
- Perform calculations
- Search the web
- Access databases
- Call APIs
- Execute code
- Interact with external systems

### 2.3 Memory System (The "Memory")

Agents need to remember:
- Previous actions and their results
- Context from earlier in the conversation
- Intermediate reasoning steps
- Final conclusions

### 2.4 Planning and Execution (The "Strategy")

Agents can:
- Break down complex tasks into subtasks
- Plan the sequence of actions needed
- Adapt plans based on new information
- Handle failures and retry with different approaches

## 3. The ReAct Framework

ReAct (Reasoning + Acting) is a framework that formalizes the agentic approach:

### 3.1 ReAct Components

**Reasoning**: The agent thinks about what it knows and what it needs to do
**Acting**: The agent takes an action using available tools
**Observing**: The agent observes the result and updates its understanding

### 3.2 Example ReAct Pattern

```
Question: "What is the population of Tokyo and how does it compare to New York?"

Thought: I need to find the population of Tokyo and New York to compare them.
Action: Search for "Tokyo population 2024"
Observation: Tokyo has a population of approximately 37.4 million people.
Thought: Now I need to find New York's population for comparison.
Action: Search for "New York City population 2024"
Observation: New York City has a population of approximately 8.8 million people.
Thought: Now I can compare these populations and provide a meaningful answer.
Action: Calculate the ratio: 37.4 / 8.8 = 4.25
Observation: Tokyo's population is about 4.25 times larger than New York's.
Answer: Tokyo has a population of approximately 37.4 million people, while New York City has about 8.8 million people. This means Tokyo's population is about 4.25 times larger than New York's.
```

## 4. Building Agents with LangChain

LangChain provides excellent support for building agentic applications. Let's explore the key components:

### 4.1 LangChain Agent Components

**Agent**: The main orchestrator that decides what to do
**Tools**: Functions that the agent can call
**Memory**: Stores conversation history and context
**LLM**: The reasoning engine (can be any LLM)

### 4.2 Basic Agent Structure

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory

# Initialize the LLM
llm = Ollama(model="phi3:mini")

# Define tools
tools = [
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="Useful for mathematical calculations"
    ),
    Tool(
        name="Search",
        func=search_function,
        description="Useful for finding current information"
    )
]

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Create the agent
agent = initialize_agent(
    tools, 
    llm, 
    agent="conversational-react-description",
    memory=memory,
    verbose=True
)
```

## 5. Types of Tools for Agents

### 5.1 Built-in Tools

**Mathematical Tools:**
- Calculator for arithmetic operations
- Statistical analysis tools
- Unit conversion tools

**Information Retrieval Tools:**
- Web search tools (DuckDuckGo, Google)
- Wikipedia search
- News search

**Data Processing Tools:**
- File reading and writing
- Data analysis (pandas, numpy)
- Visualization tools

### 5.2 Custom Tools

You can create custom tools for any specific functionality:

```python
from langchain.tools import BaseTool
from typing import Optional

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather for a city"
    
    def _run(self, city: str) -> str:
        # Implementation to get weather data
        return f"Weather in {city}: Sunny, 25°C"
    
    def _arun(self, city: str):
        raise NotImplementedError("Async not implemented")

# Usage
weather_tool = WeatherTool()
```

### 5.3 Tool Design Principles

1. **Clear Purpose**: Each tool should have a single, well-defined purpose
2. **Good Documentation**: Tools should have clear descriptions of what they do
3. **Error Handling**: Tools should handle errors gracefully
4. **Consistent Interface**: Tools should follow consistent input/output patterns

## 6. Agent Types and Architectures

### 6.1 Conversational Agents

Best for chat-based interactions:
- Maintains conversation context
- Can handle multi-turn conversations
- Good for customer service, tutoring, etc.

### 6.2 Task-Oriented Agents

Designed for specific tasks:
- Focused on completing particular goals
- May have specialized tools
- Examples: research agents, coding assistants

### 6.3 Autonomous Agents

Can operate independently:
- Set their own goals
- Plan and execute complex workflows
- Can adapt to changing circumstances

## 7. Building a Simple Research Agent

Let's walk through building a research agent that can search for information and provide summaries:

### 7.1 Agent Components

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory

# Initialize components
llm = Ollama(model="phi3:mini")
search = DuckDuckGoSearchRun()
memory = ConversationBufferMemory(memory_key="chat_history")

# Define tools
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for finding current information on the internet"
    )
]

# Create the agent
agent = initialize_agent(
    tools,
    llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True
)
```

### 7.2 Using the Agent

```python
# Ask the agent to research a topic
response = agent.run("What are the latest developments in renewable energy?")
print(response)
```

## 8. Best Practices for Agent Development

### 8.1 Tool Selection

- Choose tools that are relevant to your use case
- Ensure tools are reliable and well-tested
- Consider the cost and rate limits of external APIs

### 8.2 Prompt Engineering for Agents

- Be specific about the agent's role and capabilities
- Provide clear examples of how to use tools
- Include constraints and safety guidelines

### 8.3 Error Handling

- Implement robust error handling for tool failures
- Provide fallback mechanisms
- Log errors for debugging

### 8.4 Performance Optimization

- Cache frequently used information
- Optimize tool calls to minimize latency
- Use appropriate LLM models for your use case

## 9. Challenges and Limitations

### 9.1 Technical Challenges

**Tool Reliability:**
- External APIs may be unavailable
- Tools may return unexpected results
- Rate limiting and costs

**Reasoning Quality:**
- LLMs may make poor decisions about tool usage
- Reasoning may be inconsistent or incorrect
- Context window limitations

### 9.2 Safety and Security

**Tool Safety:**
- Tools can have side effects (file deletion, API calls)
- Need to validate inputs and outputs
- Consider security implications

**Bias and Fairness:**
- Agents may inherit biases from the underlying LLM
- Tool selection may be biased
- Need to ensure fair and ethical behavior

## 10. Future Directions

### 10.1 Advanced Agent Capabilities

- **Multi-agent systems**: Multiple agents working together
- **Hierarchical planning**: Breaking complex tasks into subtasks
- **Learning from experience**: Agents that improve over time
- **Tool creation**: Agents that can create their own tools

### 10.2 Integration with Other Technologies

- **Knowledge graphs**: For better reasoning and memory
- **Databases**: For persistent storage and retrieval
- **APIs and services**: For broader tool integration
- **IoT devices**: For physical world interaction

## 11. Conclusion

Agentic LLMs represent a significant evolution in AI capabilities, moving from passive text generation to active problem-solving and task execution. By combining the reasoning capabilities of LLMs with the power of tools and external resources, we can create systems that are more capable, flexible, and useful.

The key to building effective agents is:
1. **Clear understanding** of the problem domain
2. **Careful selection** of appropriate tools
3. **Robust implementation** with error handling
4. **Continuous testing** and improvement

In the lab session, you'll get hands-on experience building your own agentic applications, and in the assignment, you'll extend these concepts to create more sophisticated agents.

## Key Takeaways

- Agentic LLMs can reason, plan, and take actions using tools
- The ReAct framework provides a structured approach to agentic behavior
- LangChain provides excellent tools for building agents
- Tool selection and design are crucial for agent effectiveness
- Safety, reliability, and performance are important considerations

## Next Steps

- Complete the lab session to build your first agent
- Work on the assignment to create a more sophisticated agent
- Explore additional frameworks like AutoGen, CrewAI, or custom implementations
- Consider how agentic LLMs could be applied to your specific domain or interests 
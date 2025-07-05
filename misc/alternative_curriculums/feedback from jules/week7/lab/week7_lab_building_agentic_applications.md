# Lab 7: Building Your First Agentic LLM Application

**Objective:** To provide hands-on experience building agentic LLM applications using LangChain, Ollama, and various tools. You'll create agents that can reason, plan, and use tools to accomplish tasks.

**Prerequisites:**
* Python 3.x installed
* Ollama installed with a model like `phi3:mini` or `llama3:8b` pulled
* Completion of Week 6 (RAG) lab is helpful but not required
* Necessary Python libraries (install using pip):
  ```bash
  pip install langchain langchain_community langchain_community.tools duckduckgo-search
  ```

---

## Part 1: Setting Up Your Environment

### 1.1 Install Required Packages

First, let's install the necessary packages:

```bash
pip install langchain langchain_community langchain_community.tools duckduckgo-search
```

### 1.2 Verify Ollama Setup

Make sure Ollama is running and you have a model available:

```bash
# Check if Ollama is running
ollama list

# If you don't have phi3:mini, pull it
ollama pull phi3:mini
```

---

## Part 2: Building a Simple Calculator Agent

Let's start with a basic agent that can perform mathematical calculations.

### 2.1 Create the Calculator Agent

Create a new Python file called `calculator_agent.py`:

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory

def create_calculator_agent():
    """Create a simple calculator agent using Ollama and LangChain."""
    
    # Initialize the LLM
    llm = Ollama(model="phi3:mini")
    
    # Define the calculator tool
    def calculator(expression):
        """Evaluate mathematical expressions safely."""
        try:
            # Only allow basic mathematical operations for safety
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Only basic mathematical operations are allowed"
            
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"
    
    # Create tools list
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for mathematical calculations. Input should be a mathematical expression like '2 + 2' or '10 * 5'"
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
    
    return agent

def test_calculator_agent():
    """Test the calculator agent with various queries."""
    
    agent = create_calculator_agent()
    
    # Test queries
    test_queries = [
        "What is 15 + 27?",
        "Calculate 100 divided by 4",
        "What is 7 squared?",
        "If I have 3 apples and buy 5 more, how many do I have?",
        "What is the result of (10 + 5) * 2?"
    ]
    
    print("Testing Calculator Agent")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = agent.run(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    test_calculator_agent()
```

### 2.2 Run the Calculator Agent

Execute the script:

```bash
python calculator_agent.py
```

**Expected Output:** You should see the agent reasoning about each query, deciding to use the calculator tool, and providing the results.

---

## Part 3: Building a Research Agent with Web Search

Now let's create a more sophisticated agent that can search the web for information.

### 3.1 Create the Research Agent

Create a new Python file called `research_agent.py`:

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun

def create_research_agent():
    """Create a research agent that can search the web for information."""
    
    # Initialize the LLM
    llm = Ollama(model="phi3:mini")
    
    # Initialize web search tool
    search = DuckDuckGoSearchRun()
    
    # Define tools
    tools = [
        Tool(
            name="Web Search",
            func=search.run,
            description="Useful for finding current information on the internet. Use this when you need to find recent information, facts, or news about a topic."
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
    
    return agent

def test_research_agent():
    """Test the research agent with various queries."""
    
    agent = create_research_agent()
    
    # Test queries
    test_queries = [
        "What are the latest developments in renewable energy?",
        "Who won the most recent Nobel Prize in Physics?",
        "What is the current population of Tokyo?",
        "What are the main features of Python 3.12?",
        "What is the weather like in San Francisco today?"
    ]
    
    print("Testing Research Agent")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = agent.run(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    test_research_agent()
```

### 3.2 Run the Research Agent

Execute the script:

```bash
python research_agent.py
```

**Expected Output:** The agent should search the web for each query and provide informative responses based on the search results.

---

## Part 4: Building a Multi-Tool Agent

Let's create an agent that combines multiple tools for more complex tasks.

### 4.1 Create the Multi-Tool Agent

Create a new Python file called `multi_tool_agent.py`:

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
import datetime

def create_multi_tool_agent():
    """Create an agent with multiple tools: calculator, search, and time."""
    
    # Initialize the LLM
    llm = Ollama(model="phi3:mini")
    
    # Initialize web search tool
    search = DuckDuckGoSearchRun()
    
    # Define calculator tool
    def calculator(expression):
        """Evaluate mathematical expressions safely."""
        try:
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Only basic mathematical operations are allowed"
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"
    
    # Define time tool
    def get_current_time():
        """Get the current date and time."""
        now = datetime.datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Define tools
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for mathematical calculations. Input should be a mathematical expression like '2 + 2' or '10 * 5'"
        ),
        Tool(
            name="Web Search",
            func=search.run,
            description="Useful for finding current information on the internet. Use this when you need to find recent information, facts, or news about a topic."
        ),
        Tool(
            name="Current Time",
            func=get_current_time,
            description="Useful for getting the current date and time."
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
    
    return agent

def test_multi_tool_agent():
    """Test the multi-tool agent with complex queries."""
    
    agent = create_multi_tool_agent()
    
    # Test queries that might require multiple tools
    test_queries = [
        "What time is it right now?",
        "If I invest $1000 at 5% interest for 3 years, how much will I have?",
        "What is the current population of New York City and what percentage of the US population is that?",
        "What is the weather like in London today and what time is it there?",
        "How many days are there between today and Christmas 2024?"
    ]
    
    print("Testing Multi-Tool Agent")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = agent.run(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    test_multi_tool_agent()
```

### 4.2 Run the Multi-Tool Agent

Execute the script:

```bash
python multi_tool_agent.py
```

**Expected Output:** The agent should intelligently choose which tools to use for each query, sometimes using multiple tools to provide comprehensive answers.

---

## Part 5: Creating a Custom Tool

Let's create a custom tool for a specific use case.

### 5.1 Create a Custom Weather Tool

Create a new Python file called `custom_weather_agent.py`:

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from typing import Optional
import random

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather information for a city. Input should be a city name."
    
    def _run(self, city: str) -> str:
        """Get weather information for a city (simulated)."""
        # This is a simulated weather tool
        # In a real application, you would call a weather API
        weather_conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Partly Cloudy"]
        temperatures = list(range(10, 35))  # 10-35 degrees Celsius
        
        condition = random.choice(weather_conditions)
        temperature = random.choice(temperatures)
        
        return f"Weather in {city}: {condition}, {temperature}°C"
    
    def _arun(self, city: str):
        raise NotImplementedError("Async not implemented")

def create_weather_agent():
    """Create an agent with a custom weather tool."""
    
    # Initialize the LLM
    llm = Ollama(model="phi3:mini")
    
    # Create custom weather tool
    weather_tool = WeatherTool()
    
    # Define tools
    tools = [
        Tool(
            name="Weather",
            func=weather_tool._run,
            description="Get current weather information for a city. Input should be a city name like 'New York' or 'Tokyo'."
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
    
    return agent

def test_weather_agent():
    """Test the weather agent."""
    
    agent = create_weather_agent()
    
    # Test queries
    test_queries = [
        "What's the weather like in Paris?",
        "How's the weather in Tokyo today?",
        "Tell me about the weather in New York",
        "What's the current weather in London?"
    ]
    
    print("Testing Weather Agent")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = agent.run(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    test_weather_agent()
```

### 5.2 Run the Custom Weather Agent

Execute the script:

```bash
python custom_weather_agent.py
```

---

## Part 6: Interactive Agent Session

Now let's create an interactive session where you can chat with an agent.

### 6.1 Create Interactive Agent

Create a new Python file called `interactive_agent.py`:

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
import datetime

def create_interactive_agent():
    """Create an interactive agent with multiple tools."""
    
    # Initialize the LLM
    llm = Ollama(model="phi3:mini")
    
    # Initialize web search tool
    search = DuckDuckGoSearchRun()
    
    # Define calculator tool
    def calculator(expression):
        """Evaluate mathematical expressions safely."""
        try:
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Only basic mathematical operations are allowed"
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"
    
    # Define time tool
    def get_current_time():
        """Get the current date and time."""
        now = datetime.datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Define tools
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for mathematical calculations. Input should be a mathematical expression like '2 + 2' or '10 * 5'"
        ),
        Tool(
            name="Web Search",
            func=search.run,
            description="Useful for finding current information on the internet. Use this when you need to find recent information, facts, or news about a topic."
        ),
        Tool(
            name="Current Time",
            func=get_current_time,
            description="Useful for getting the current date and time."
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
    
    return agent

def interactive_session():
    """Run an interactive session with the agent."""
    
    agent = create_interactive_agent()
    
    print("Welcome to the Interactive Agent Session!")
    print("This agent can help you with calculations, web searches, and time queries.")
    print("Type 'quit' to exit the session.")
    print("=" * 60)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for quit command
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Agent: Goodbye! Thanks for chatting with me!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Get agent response
            print("\nAgent is thinking...")
            response = agent.run(user_input)
            print(f"Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nAgent: Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    interactive_session()
```

### 6.2 Run the Interactive Session

Execute the script:

```bash
python interactive_agent.py
```

Try asking the agent various questions like:
- "What is 25 * 4?"
- "What are the latest news about AI?"
- "What time is it right now?"
- "If I have $500 and spend $150, how much do I have left?"
- "What is the population of Tokyo?"

---

## Part 7: Analysis and Reflection

After completing the lab exercises, take some time to reflect on your experience:

### 7.1 Questions to Consider

1. **Tool Selection**: How did the agent decide which tools to use for different queries?
2. **Reasoning Quality**: How well did the agent reason about when to use each tool?
3. **Response Quality**: How accurate and helpful were the agent's responses?
4. **Limitations**: What limitations did you observe in the agent's capabilities?

### 7.2 Observations to Document

Document your observations about:
- Which queries worked well and which didn't
- How the agent handled complex queries requiring multiple tools
- The quality of reasoning and decision-making
- Any errors or unexpected behaviors

### 7.3 Potential Improvements

Think about how you could improve the agents:
- What additional tools would be useful?
- How could the reasoning be improved?
- What safety measures should be added?
- How could the user experience be enhanced?

---

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**: Make sure Ollama is running and the model is available
2. **Import Errors**: Ensure all required packages are installed
3. **Tool Errors**: Check that tool functions handle errors gracefully
4. **Memory Issues**: Some models may be too large for your system

### Performance Tips

1. Use smaller models like `phi3:mini` for faster responses
2. Limit the number of tools to reduce complexity
3. Add error handling to prevent crashes
4. Consider caching frequently used information

---

## Next Steps

After completing this lab:

1. **Assignment**: Work on the Week 7 assignment to extend your agent
2. **Experimentation**: Try creating agents with different combinations of tools
3. **Custom Tools**: Build your own custom tools for specific use cases
4. **Advanced Frameworks**: Explore other agent frameworks like AutoGen or CrewAI

This lab provides a solid foundation for building agentic LLM applications. The skills you've learned here can be applied to create more sophisticated agents for various domains and use cases. 
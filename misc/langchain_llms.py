from langchain_anthropic import ChatAnthropic
from langchain_openai    import ChatOpenAI
from langchain_ollama    import ChatOllama

llm = ChatAnthropic(model = "claude-3-opus-20240229")
llm = ChatOpenAI   (model = "gpt-4o")
llm = ChatOllama   (model = "llama3.2")

result = llm.invoke("Tell me a joke about programming")
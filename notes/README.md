# BeeAI Framework Basics: A Full small Tutorial

In the following sections, we’ll dive deep into the BeeAI framework—from prompt templating to building complete ReAct agents and workflows. This comprehensive tutorial provides all the Python code you need to build robust, multi-step AI agents.

---

## 1. Prompt Templates

One of the core constructs in BeeAI is the **PromptTemplate**. It allows you to dynamically insert data into a prompt before sending it to a language model. BeeAI uses the Mustache templating language for formatting.

### Example: Creating a RAG Template

```python
from pydantic import BaseModel
from beeai_framework.utils.templates import PromptTemplate

# Defines the structure of the input data that can be passed to the template
class RAGTemplateInput(BaseModel):
    question: str
    context: str

# Define the prompt template
rag_template: PromptTemplate = PromptTemplate(
    schema=RAGTemplateInput,
    template="""
Context: {{context}}
Question: {{question}}

Provide a concise answer based on the context. Avoid statements such as 'Based on the context' or 'According to the context' etc. """,
)

# Render the template using an instance of the input model
prompt = rag_template.render(
    RAGTemplateInput(
        question="What is the capital of France?",
        context="France is a country in Europe. Its capital city is Paris, known for its culture and history.",
    )
)

# Print the rendered prompt
print(prompt)
```

---

## 2. More Complex Templates

You can also create templates that include lists or iterate over data. The following example shows a template that iterates over search results.

```python
from pydantic import BaseModel
from beeai_framework.utils.templates import PromptTemplate

# Individual search result schema
class SearchResult(BaseModel):
    title: str
    url: str
    content: str

# Input specification for the template
class SearchTemplateInput(BaseModel):
    question: str
    results: list[SearchResult]

# Define the template; it will iterate over the results list
search_template: PromptTemplate = PromptTemplate(
    schema=SearchTemplateInput,
    template="""
Search results:
{{#results.0}}
{{#results}}
Title: {{title}}
Url: {{url}}
Content: {{content}}
{{/results}}
{{/results.0}}

Question: {{question}}
Provide a concise answer based on the search results provided.""",
)

# Render the template using an instance of the input model
prompt = search_template.render(
    SearchTemplateInput(
        question="What is the capital of France?",
        results=[
            SearchResult(
                title="France",
                url="https://en.wikipedia.org/wiki/France",
                content="France is a country in Europe. Its capital city is Paris, known for its culture and history.",
            )
        ],
    )
)

# Print the rendered prompt
print(prompt)
```

---

## 3. The ChatModel

Once you have your templates, you can interact with a language model. BeeAI supports a variety of LLMs via the **ChatModel** interface.

### Example: Sending a User Message

```python
from beeai_framework.backend.message import UserMessage

# Create a user message to start a chat with the model
user_message = UserMessage(content="Hello! Can you tell me what is the capital of France?")
```

### Example: Interacting with the Model

```python
from beeai_framework.backend.chat import ChatModel, ChatModelInput, ChatModelOutput

# Create a ChatModel (using IBM Granite 3.1 8B via Ollama in this example)
model = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Send the user message to the model
output: ChatModelOutput = await model.create(ChatModelInput(messages=[user_message]))

print(output.get_text_content())
```

---

## 4. Memory: Storing Conversation History

Memory in BeeAI is a convenient way to store the conversation history.

```python
from beeai_framework.backend.message import AssistantMessage
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

memory = UnconstrainedMemory()

await memory.add_many(
    [
        user_message,
        AssistantMessage(content=output.get_text_content()),
        UserMessage(content="If you had to recommend one thing to do there, what would it be?"),
    ]
)

output: ChatModelOutput = await model.create(ChatModelInput(messages=memory.messages))
print(output.get_text_content())
```

---

## 5. Combining Templates and Messages

You can render a prompt template and then use it as the content of a message to the ChatModel.

```python
# Some context for the model
context = """The geography of Ireland comprises relatively low-lying mountains surrounding a central plain, with several navigable rivers extending inland.
Its lush vegetation is a product of its mild but changeable climate which is free of extremes in temperature.
Much of Ireland was woodland until the end of the Middle Ages. Today, woodland makes up about 10% of the island,
compared with a European average of over 33%, with most of it being non-native conifer plantations.
The Irish climate is influenced by the Atlantic Ocean and thus very moderate, and winters are milder than expected for such a northerly area,
although summers are cooler than those in continental Europe. Rainfall and cloud cover are abundant.
"""

# Reuse our RAG template
prompt = rag_template.render(RAGTemplateInput(question="How much of Ireland is forested?", context=context))

output: ChatModelOutput = await model.create(ChatModelInput(messages=[UserMessage(content=prompt)]))
print(output.get_text_content())
```

---

## 6. Structured Outputs

To have the LLM output data in a specific format, you can use structured output definitions.

```python
from typing import Literal
from pydantic import BaseModel, Field
from beeai_framework.backend.chat import ChatModelStructureInput

# Define the output structure
class CharacterSchema(BaseModel):
    name: str = Field(description="The name of the character.")
    occupation: str = Field(description="The occupation of the character.")
    species: Literal["Human", "Insectoid", "Void-Serpent", "Synth", "Ethereal", "Liquid-Metal"] = Field(
        description="The race of the character."
    )
    back_story: str = Field(description="Brief backstory of this character.")

user_message = UserMessage("Create a fantasy sci-fi character for my new game. This character will be the main protagonist, be creative.")

response = await model.create_structure(ChatModelStructureInput(schema=CharacterSchema, messages=[user_message]))
print(response.object)
```

---

## 7. System Prompts

System messages can instruct the model to adopt a specific style or behavior.

```python
from beeai_framework.backend.message import SystemMessage

system_message = SystemMessage(content="You are pirate. You always respond using pirate slang.")
user_message = UserMessage(content="What is a baby hedgehog called?")
output: ChatModelOutput = await model.create(ChatModelInput(messages=[system_message, user_message]))
print(output.get_text_content())
```

---

## 8. Building an Agent with BeeAI

Now that you have seen how to interact with the ChatModel, here’s how to build an agent using the BeeAI framework.

```python
from typing import Any
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput, BeeRunOutput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.emitter.types import EmitterOptions
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Construct the ChatModel instance
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Construct the agent instance with no external tools
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[], memory=UnconstrainedMemory()))

async def process_agent_events(event_data: dict[str, Any], event_meta: EventMeta) -> None:
    """Process agent events and log appropriately"""
    if event_meta.name == "error":
        print("Agent 🤖 : ", event_data["error"])
    elif event_meta.name == "retry":
        print("Agent 🤖 : ", "retrying the action...")
    elif event_meta.name == "update":
        print(f"Agent({event_data['update']['key']}) 🤖 : ", event_data["update"]["parsedValue"])

# Observer to capture agent events
async def observer(emitter: Emitter) -> None:
    emitter.on("*.*", process_agent_events, EmitterOptions(match_nested=True))

# Run the agent
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What chemical elements make up a water molecule?")
).observe(observer)
```

---

## 9. Using Tools with the Agent

BeeAI allows agents to be extended with tools. For example, you can add a weather forecast lookup tool.

### Using a Built-in Tool (OpenMeteoTool)

```python
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Create an agent and add the OpenMeteoTool for weather queries
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[OpenMeteoTool()], memory=UnconstrainedMemory()))

# Run the agent to get the current weather in London
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What's the current weather in London?")
).observe(observer)
```

---

### Integrating External Tools: Wikipedia via LangChain

You can also import tools from other libraries. For instance, integrating Wikipedia search functionality from LangChain:

```python
from typing import Any
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.tools import Tool
from beeai_framework.tools.tool import StringToolOutput

class LangChainWikipediaToolInput(BaseModel):
    query: str = Field(description="The topic or question to search for on Wikipedia.")

class LangChainWikipediaTool(Tool):
    """Adapter class to integrate LangChain's Wikipedia tool with our framework"""
    name = "Wikipedia"
    description = "Search factual and historical information from Wikipedia about given topics."
    input_schema = LangChainWikipediaToolInput

    def __init__(self) -> None:
        super().__init__()
        wikipedia = WikipediaAPIWrapper()
        self.wikipedia = WikipediaQueryRun(api_wrapper=wikipedia)

    def _run(self, input: LangChainWikipediaToolInput, _: Any | None = None) -> None:
        query = input.query
        try:
            result = self.wikipedia.run(query)
            return StringToolOutput(result=result)
        except Exception as e:
            print(f"Wikipedia search error: {e!s}")
            return f"Error searching Wikipedia: {e!s}"

chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[LangChainWikipediaTool()], memory=UnconstrainedMemory()))

result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="Who is the current president of the European Commission?")
).observe(observer)
```

#### A Shorter Form Using the `@tool` Decorator

```python
from langchain_community.tools import WikipediaQueryRun  # noqa: F811
from langchain_community.utilities import WikipediaAPIWrapper  # noqa: F811
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.tools import Tool, tool

@tool
def langchain_wikipedia_tool(expression: str) -> str:
    """
    Search factual and historical information, including biography, history, politics, geography, society, culture,
    science, technology, people, animal species, mathematics, and other subjects.

    Args:
        expression: The topic or question to search for on Wikipedia.

    Returns:
        The information found via searching Wikipedia.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return StringToolOutput(wikipedia.run(expression))

chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[langchain_wikipedia_tool], memory=UnconstrainedMemory()))

result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What is the longest living vertebrate?")
).observe(observer)
```

---

## 10. BeeAI Workflows

Workflows let you combine multiple steps and build complex agents. They define a state and a series of steps (functions) that update that state.

### Basics of Workflows

```python
import traceback
from pydantic import BaseModel, ValidationError
from beeai_framework.workflows.workflow import Workflow, WorkflowError

# Define a global state model
class MessageState(BaseModel):
    message: str

# A simple workflow step that modifies the state
async def my_first_step(state: MessageState) -> None:
    state.message += " World"  # Modify the state
    print("Running first step!")
    return Workflow.END

try:
    # Define the workflow graph
    basic_workflow = Workflow(schema=MessageState, name="MyWorkflow")
    basic_workflow.add_step("my_first_step", my_first_step)
    
    # Execute the workflow
    basic_response = await basic_workflow.run(MessageState(message="Hello"))
    print("State after workflow run:", basic_response.state)
except WorkflowError:
    traceback.print_exc()
except ValidationError:
    traceback.print_exc()
```

---

### A Multi-Step Workflow with Tools

This example builds a simple web search agent that generates a query, runs a search, and produces an answer based on the results.

```python
from langchain_community.utilities import SearxSearchWrapper
from pydantic import Field
from beeai_framework.backend.chat import ChatModel, ChatModelOutput, ChatModelStructureOutput
from beeai_framework.backend.message import UserMessage
from beeai_framework.utils.templates import PromptTemplate

# Define the workflow state
class SearchAgentState(BaseModel):
    question: str
    search_results: str | None = None
    answer: str | None = None

# Create a ChatModel instance
model = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Set up the web search tool (ensure your SearXNG instance is running)
search_tool = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")

# Define prompt templates and schemas
class QuestionInput(BaseModel):
    question: str

class SearchRAGInput(BaseModel):
    question: str
    search_results: str

search_query_template = PromptTemplate(
    schema=QuestionInput,
    template="""Convert the following question into a concise, effective web search query using keywords and operators for accuracy.
Question: {{question}}""",
)

search_rag_template = PromptTemplate(
    schema=SearchRAGInput,
    template="""Search results:
{{search_results}}

Question: {{question}}
Provide a concise answer based on the search results provided. If the results are irrelevant or insufficient, say 'I don't know.' Avoid phrases such as 'According to the results...'.""",
)

# Structured output schema for the search query
class WebSearchQuery(BaseModel):
    query: str = Field(description="The web search query.")

# Step 1: Generate a search query and run a web search
async def web_search(state: SearchAgentState) -> str:
    print("Step: ", "web_search")
    prompt = search_query_template.render(QuestionInput(question=state.question))
    response: ChatModelStructureOutput = await model.create_structure(
        {"schema": WebSearchQuery, "messages": [UserMessage(prompt)]}
    )
    state.search_results = search_tool.run(response.object["query"])
    return "generate_answer"

# Step 2: Generate an answer based on the search results
async def generate_answer(state: SearchAgentState) -> str:
    print("Step: ", "generate_answer")
    prompt = search_rag_template.render(
        SearchRAGInput(
            question=state.question, 
            search_results=state.search_results or "No results available."
        )
    )
    output: ChatModelOutput = await model.create({"messages": [UserMessage(prompt)]})
    state.answer = output.get_text_content()
    return Workflow.END

try:
    # Define the workflow graph
    search_agent_workflow = Workflow(schema=SearchAgentState, name="WebSearchAgent")
    search_agent_workflow.add_step("web_search", web_search)
    search_agent_workflow.add_step("generate_answer", generate_answer)

    # Execute the workflow
    search_response = await search_agent_workflow.run(
        SearchAgentState(question="What is the term for a baby hedgehog?")
    )

    print("*****")
    print("Question: ", search_response.state.question)
    print("Answer: ", search_response.state.answer)
except WorkflowError:
    traceback.print_exc()
except ValidationError:
    traceback.print_exc()
```

---

## 11. Adding Memory to a Workflow Agent

To support interactive conversations, you can add memory to your workflow agent. This example shows how to integrate message history.

```python
from pydantic import InstanceOf
from beeai_framework.backend.message import AssistantMessage, SystemMessage
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Define the chat workflow state with memory
class ChatState(BaseModel):
    memory: InstanceOf[UnconstrainedMemory]
    output: str | None = None

async def chat(state: ChatState) -> str:
    output: ChatModelOutput = await model.create({"messages": state.memory.messages})
    state.output = output.get_text_content()
    return Workflow.END

memory = UnconstrainedMemory()
await memory.add(SystemMessage(content="You are a helpful and friendly AI assistant."))

try:
    # Define the chat workflow
    chat_workflow = Workflow(ChatState)
    chat_workflow.add_step("chat", chat)
    chat_workflow.add_step("generate_answer", generate_answer)

    while True:
        user_input = input("User (type 'exit' to stop): ")
        if user_input == "exit":
            break
        await memory.add(UserMessage(content=user_input))
        response = await chat_workflow.run(ChatState(memory=memory))
        await memory.add(AssistantMessage(content=response.state.output))
        print("Assistant: ", response.state.output)
except WorkflowError:
    traceback.print_exc()
except ValidationError:
    traceback.print_exc()
```

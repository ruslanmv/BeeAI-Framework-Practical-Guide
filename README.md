# BeeAI Framework: Your Guide from Zero to Hero

Welcome to the BeeAI Framework! This repository is your comprehensive guide to learning and using BeeAI, taking you from an absolute beginner to a proficient developer. In this extended tutorial we cover:

- **Prompt Templates:** How to create and render templates for dynamic prompt generation.
- **ChatModel Interaction:** How to interact with a language model using message-based inputs.
- **Memory Handling:** How to build a conversation history for context.
- **Structured Outputs:** Enforcing output formats with Pydantic schemas.
- **System Prompts:** Guiding LLM behavior with system messages.
- **ReAct Agents and Tools:** Building agents that can reason and act, including integration with external tools.
- **Workflows:** Combining all of the above into a multi-step process, including adding memory.

Below you’ll find all the code examples along with explanations.

---

## BeeAI Framework Basics

These examples demonstrate the fundamental usage patterns of BeeAI in Python. They progressively increase in complexity, providing a well-rounded overview of the framework.

---

### 1. Prompt Templates

One of the core constructs in the BeeAI framework is the `PromptTemplate`. It allows you to dynamically insert data into a prompt before sending it to a language model. BeeAI uses the Mustache templating language for prompt formatting.

#### Example: RAG Prompt Template

```python
from pydantic import BaseModel
from beeai_framework.utils.templates import PromptTemplate

# Define the structure of the input data that will be passed to the template.
class RAGTemplateInput(BaseModel):
    question: str
    context: str

# Define the prompt template.
rag_template: PromptTemplate = PromptTemplate(
    schema=RAGTemplateInput,
    template="""
Context: {{context}}
Question: {{question}}

Provide a concise answer based on the context. Avoid statements such as 'Based on the context' or 'According to the context' etc. """,
)

# Render the template using an instance of the input model.
prompt = rag_template.render(
    RAGTemplateInput(
        question="What is the capital of France?",
        context="France is a country in Europe. Its capital city is Paris, known for its culture and history.",
    )
)

# Print the rendered prompt.
print(prompt)
```

---

### 2. More Complex Templates

The `PromptTemplate` class also supports more complex structures. For example, you can iterate over a list of search results to build a prompt.

#### Example: Template with a List of Search Results

```python
from pydantic import BaseModel
from beeai_framework.utils.templates import PromptTemplate

# Individual search result schema.
class SearchResult(BaseModel):
    title: str
    url: str
    content: str

# Input specification for the template.
class SearchTemplateInput(BaseModel):
    question: str
    results: list[SearchResult]

# Define the template that iterates over the search results.
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

# Render the template with sample data.
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

# Print the rendered prompt.
print(prompt)
```

---

### 3. The ChatModel

Once you have your prompt templates set up, you can begin interacting with a language model. BeeAI supports various LLMs through the `ChatModel` interface.

#### Example: Creating a User Message

```python
from beeai_framework.backend.message import UserMessage

# Create a user message to start a chat with the model.
user_message = UserMessage(content="Hello! Can you tell me what is the capital of France?")
```

#### Example: Sending a Message to the ChatModel

```python
from beeai_framework.backend.chat import ChatModel, ChatModelInput, ChatModelOutput

# Create a ChatModel instance that interfaces with Granite 3.1 (via Ollama).
model = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Send the user message and get the model's response.
output: ChatModelOutput = await model.create(ChatModelInput(messages=[user_message]))

# Print the model's response.
print(output.get_text_content())
```

---

### 4. Memory Handling

Memory is a convenient way to store the conversation history (a series of messages) that the model uses for context.

#### Example: Storing and Retrieving Conversation History

```python
from beeai_framework.backend.message import AssistantMessage
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Create an unconstrained memory instance.
memory = UnconstrainedMemory()

# Add a series of messages to the memory.
await memory.add_many(
    [
        user_message,
        AssistantMessage(content=output.get_text_content()),
        UserMessage(content="If you had to recommend one thing to do there, what would it be?"),
    ]
)

# Send the complete message history to the model.
output: ChatModelOutput = await model.create(ChatModelInput(messages=memory.messages))
print(output.get_text_content())
```

---

### 5. Combining Templates and Messages

You can render a prompt from a template and then send it as a message to the ChatModel.

#### Example: Rendering a Template and Sending as a Message

```python
# Some context that the model will use (e.g., from Wikipedia on Ireland).
context = """The geography of Ireland comprises relatively low-lying mountains surrounding a central plain, with several navigable rivers extending inland.
Its lush vegetation is a product of its mild but changeable climate which is free of extremes in temperature.
Much of Ireland was woodland until the end of the Middle Ages. Today, woodland makes up about 10% of the island,
compared with a European average of over 33%, with most of it being non-native conifer plantations.
The Irish climate is influenced by the Atlantic Ocean and thus very moderate, and winters are milder than expected for such a northerly area,
although summers are cooler than those in continental Europe. Rainfall and cloud cover are abundant.
"""

# Reuse the previously defined RAG template.
prompt = rag_template.render(RAGTemplateInput(question="How much of Ireland is forested?", context=context))

# Send the rendered prompt to the model.
output: ChatModelOutput = await model.create(ChatModelInput(messages=[UserMessage(content=prompt)]))
print(output.get_text_content())
```

---

### 6. Structured Outputs

Sometimes you want the LLM to generate output in a specific format. You can enforce this using structured outputs with a Pydantic schema.

#### Example: Enforcing a Specific Output Format

```python
from typing import Literal
from pydantic import Field
from beeai_framework.backend.chat import ChatModelStructureInput

# Define the output structure for a character.
class CharacterSchema(BaseModel):
    name: str = Field(description="The name of the character.")
    occupation: str = Field(description="The occupation of the character.")
    species: Literal["Human", "Insectoid", "Void-Serpent", "Synth", "Ethereal", "Liquid-Metal"] = Field(
        description="The race of the character."
    )
    back_story: str = Field(description="Brief backstory of this character.")

# Create a user message instructing the model to generate a character.
user_message = UserMessage(
    "Create a fantasy sci-fi character for my new game. This character will be the main protagonist, be creative."
)

# Request a structured response from the model.
response = await model.create_structure(ChatModelStructureInput(schema=CharacterSchema, messages=[user_message]))
print(response.object)
```

---

### 7. System Prompts

System messages can guide the overall behavior of the language model.

#### Example: Using a System Message

```python
from beeai_framework.backend.message import SystemMessage

# Create a system message that instructs the LLM to respond like a pirate.
system_message = SystemMessage(content="You are pirate. You always respond using pirate slang.")

# Create a new user message.
user_message = UserMessage(content="What is a baby hedgehog called?")

# Send both messages to the model.
output: ChatModelOutput = await model.create(ChatModelInput(messages=[system_message, user_message]))
print(output.get_text_content())
```

---

## BeeAI ReAct Agents

The BeeAI ReAct agent implements the “Reasoning and Acting” pattern, separating the process into distinct steps. This section shows how to build an agent that uses its own memory for reasoning and even integrates tools for added functionality.

### 1. Basic ReAct Agent

#### Example: Setting Up a Basic ReAct Agent

```python
from typing import Any
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput, BeeRunOutput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.emitter.types import EmitterOptions
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Construct the BeeAgent without external tools.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[], memory=UnconstrainedMemory()))

# Define a function to process agent events.
async def process_agent_events(event_data: dict[str, Any], event_meta: EventMeta) -> None:
    if event_meta.name == "error":
        print("Agent 🤖 : ", event_data["error"])
    elif event_meta.name == "retry":
        print("Agent 🤖 : ", "retrying the action...")
    elif event_meta.name == "update":
        print(f"Agent({event_data['update']['key']}) 🤖 : ", event_data["update"]["parsedValue"])

# Attach an observer to log events.
async def observer(emitter: Emitter) -> None:
    emitter.on("*.*", process_agent_events, EmitterOptions(match_nested=True))

# Run the agent with a sample prompt.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What chemical elements make up a water molecule?")
).observe(observer)
```

---

### 2. Using Tools with the Agent

Agents can be extended with tools so that they can perform external actions, like fetching weather data.

#### Example: Using a Built-In Weather Tool

```python
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Create an agent that includes the OpenMeteoTool.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[OpenMeteoTool()], memory=UnconstrainedMemory()))

# Run the agent with a prompt about the weather.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What's the current weather in London?")
).observe(observer)
```

---

### 3. Imported Tools

You can also import tools from other libraries. Below are two examples that show how to integrate Wikipedia search via LangChain.

#### Example: Long-Form Integration with Wikipedia

```python
from typing import Any
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.tools import Tool
from beeai_framework.tools.tool import StringToolOutput

# Define the input schema for the Wikipedia tool.
class LangChainWikipediaToolInput(BaseModel):
    query: str = Field(description="The topic or question to search for on Wikipedia.")

# Adapter class to integrate LangChain's Wikipedia tool.
class LangChainWikipediaTool(Tool):
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

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Create an agent that uses the custom Wikipedia tool.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[LangChainWikipediaTool()], memory=UnconstrainedMemory()))

# Run the agent with a query about the European Commission.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="Who is the current president of the European Commission?")
).observe(observer)
```

#### Example: Shorter Form Using the `@tool` Decorator

```python
from langchain_community.tools import WikipediaQueryRun  # noqa: F811
from langchain_community.utilities import WikipediaAPIWrapper  # noqa: F811
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.tools import Tool, tool

# Define a tool using the decorator.
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

# Create a ChatModel instance.
chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Create an agent that uses the decorated Wikipedia tool.
agent = BeeAgent(bee_input=BeeInput(llm=chat_model, tools=[langchain_wikipedia_tool], memory=UnconstrainedMemory()))

# Run the agent with a query about the longest living vertebrate.
result: BeeRunOutput = await agent.run(
    run_input=BeeRunInput(prompt="What is the longest living vertebrate?")
).observe(observer)
```

---

## BeeAI Workflows

Workflows allow you to combine what you’ve learned into a coherent multi-step process. A workflow is defined by a state (a Pydantic model) and steps (Python functions) that update the state and determine the next step.

### 1. Basics of Workflows

#### Example: A Simple One-Step Workflow

```python
import traceback
from pydantic import BaseModel, ValidationError
from beeai_framework.workflows.workflow import Workflow, WorkflowError

# Define the global state for the workflow.
class MessageState(BaseModel):
    message: str

# Define a workflow step.
async def my_first_step(state: MessageState) -> None:
    state.message += " World"  # Modify the state.
    print("Running first step!")
    return Workflow.END  # Signal the end of the workflow.

try:
    # Create the workflow.
    basic_workflow = Workflow(schema=MessageState, name="MyWorkflow")
    basic_workflow.add_step("my_first_step", my_first_step)

    # Run the workflow.
    basic_response = await basic_workflow.run(MessageState(message="Hello"))
    print("State after workflow run:", basic_response.state)

except WorkflowError:
    traceback.print_exc()
except ValidationError:
    traceback.print_exc()
```

---

### 2. A Multi-Step Workflow with Tools

This example builds a web search agent that:
- Generates a search query from a question.
- Uses a web search tool to fetch results.
- Uses the search results to generate an answer.

#### Setup: Import and State Definition

```python
from langchain_community.utilities import SearxSearchWrapper
from pydantic import Field
from beeai_framework.backend.chat import ChatModel, ChatModelOutput, ChatModelStructureOutput
from beeai_framework.backend.message import UserMessage
from beeai_framework.utils.templates import PromptTemplate

# Define the workflow state.
class SearchAgentState(BaseModel):
    question: str
    search_results: str | None = None
    answer: str | None = None
```

#### Create the ChatModel and Web Search Tool

```python
# Create a ChatModel instance.
model = ChatModel.from_name("ollama:granite3.1-dense:8b")

# Create a web search tool (requires a running local SearXNG instance).
search_tool = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")
```

#### Define Templates and Output Schema

```python
# Input schemas for the templates.
class QuestionInput(BaseModel):
    question: str

class SearchRAGInput(BaseModel):
    question: str
    search_results: str

# Template to convert a question into a search query.
search_query_template = PromptTemplate(
    schema=QuestionInput,
    template="""Convert the following question into a concise, effective web search query using keywords and operators for accuracy.
Question: {{question}}""",
)

# Template to generate an answer from search results.
search_rag_template = PromptTemplate(
    schema=SearchRAGInput,
    template="""Search results:
{{search_results}}

Question: {{question}}
Provide a concise answer based on the search results provided. If the results are irrelevant or insufficient, say 'I don't know.' Avoid phrases such as 'According to the results...'.""",
)

# Define the structured output for the search query.
class WebSearchQuery(BaseModel):
    query: str = Field(description="The web search query.")
```

#### Define Workflow Steps

**Step 1: Web Search**

```python
async def web_search(state: SearchAgentState) -> str:
    print("Step: ", "web_search")
    # Generate a search query using the template.
    prompt = search_query_template.render(QuestionInput(question=state.question))
    response: ChatModelStructureOutput = await model.create_structure(
        {
            "schema": WebSearchQuery,
            "messages": [UserMessage(prompt)],
        }
    )
    # Run the web search and store the results.
    state.search_results = search_tool.run(response.object["query"])
    return "generate_answer"  # Transition to the next step.
```

**Step 2: Generate Answer**

```python
async def generate_answer(state: SearchAgentState) -> str:
    print("Step: ", "generate_answer")
    # Generate an answer using the search results.
    prompt = search_rag_template.render(
        SearchRAGInput(question=state.question, search_results=state.search_results or "No results available.")
    )
    output: ChatModelOutput = await model.create({"messages": [UserMessage(prompt)]})
    state.answer = output.get_text_content()  # Store the answer.
    return Workflow.END  # End the workflow.
```

#### Define and Run the Workflow

```python
import traceback

try:
    # Define the workflow and add steps.
    search_agent_workflow = Workflow(schema=SearchAgentState, name="WebSearchAgent")
    search_agent_workflow.add_step("web_search", web_search)
    search_agent_workflow.add_step("generate_answer", generate_answer)

    # Execute the workflow with a sample question.
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

### 3. Adding Memory to a Workflow Agent

To create a conversational agent that remembers previous interactions, you can integrate memory into your workflow.

#### Example: A Chat Workflow with Memory

```python
from pydantic import InstanceOf
from beeai_framework.backend.message import AssistantMessage, SystemMessage
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Define a state that holds a memory instance.
class ChatState(BaseModel):
    memory: InstanceOf[UnconstrainedMemory]
    output: str | None = None

# Define a workflow step that sends all messages in memory to the model.
async def chat(state: ChatState) -> str:
    output: ChatModelOutput = await model.create({"messages": state.memory.messages})
    state.output = output.get_text_content()
    return Workflow.END

# Create a memory instance and add an initial system message.
memory = UnconstrainedMemory()
await memory.add(SystemMessage(content="You are a helpful and friendly AI assistant."))

try:
    # Define a workflow that uses memory.
    chat_workflow = Workflow(ChatState)
    chat_workflow.add_step("chat", chat)
    chat_workflow.add_step("generate_answer", generate_answer)  # Reusing our previous answer generator.

    # Run an interactive loop.
    while True:
        user_input = input("User (type 'exit' to stop): ")
        if user_input == "exit":
            break
        # Add the user's message to memory.
        await memory.add(UserMessage(content=user_input))
        # Run the workflow with the current memory.
        response = await chat_workflow.run(ChatState(memory=memory))
        # Add the assistant's response to memory.
        await memory.add(AssistantMessage(content=response.state.output))
        print("Assistant: ", response.state.output)

except WorkflowError:
    traceback.print_exc()
except ValidationError:
    traceback.print_exc()
```

---

## Next Steps

Now that you have seen how to:
- Create prompt templates and render them dynamically.
- Interact with language models using ChatModel.
- Maintain conversation history with memory.
- Build structured output responses.
- Build a multi-step workflow (with and without memory).
- Configure a ReAct agent with custom and imported tools.



## Contact

For questions, discussions, or support, reach out to us via:

  * Email: [contact@ruslanmv.com](mailto:contact@ruslanmv.com)
  * GitHub Discussions: [BeeAI Framework Discussions](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/your-username/BeeAI-Framework-Practical-Guide/discussions)
  * Community Forum: [BeeAI Community](https://www.google.com/url?sa=E&source=gmail&q=https://community.beeai.org)

## Acknowledgements

We sincerely thank our contributors, researchers, and supporters who have helped shape BeeAI. Special thanks to the open-source community for their invaluable feedback and contributions\!

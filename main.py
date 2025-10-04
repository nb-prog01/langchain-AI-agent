from dotenv import load_dotenv
import re
import json
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.globals import set_debug
from tools import search_tool, wiki_tool, save_tool, save_to_txt

# load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path=".env.local")

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources:list[str]
    tools_used:list[str]

llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", verbose=True)
# response=llm.invoke("what is meaning of life?")
# print(response)

# llm=ChatAnthropic(model="claude-sonnet-4-5-20250929")

parser=PydanticOutputParser(pydantic_object=ResearchResponse)

prompt=ChatPromptTemplate.from_messages([
    ("system",
     """
     You are an expert research assistant. You will be given a topic to research. Use the tools at your disposal to gather information and provide a concise summary of the topic along with sources.
    Research the user query. Use the tools as needed and provide a summary, sources, and tools used in your response. Format your response as per the provided schema.
    Use the following format:\n{format_instructions}
    """,
    ),
    ("placeholder", "{chat_history}"),
    ("human", "Research the following query: {query}"),
    ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools=[search_tool, wiki_tool, save_tool]
agent=create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

set_debug(True)

agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True)
query=input("Enter a research Query: ")
raw_response=agent_executor.invoke({"query":query})

output_text = raw_response.get("output", "")

print("\n--- RAW MODEL OUTPUT ---")
print(output_text)

# Extract JSON block (inside ```json ... ```)
json_pattern = r"```json\s*(.*?)\s*```"
match = re.search(json_pattern, output_text, re.DOTALL)

if not match:
    print("No JSON block found.")
else:
    json_str = match.group(1).strip()

    try:
        parsed = json.loads(json_str)
        print("\nParsed JSON:")
        print(json.dumps(parsed, indent=2))

        # Check for explicit save intent
        should_save = False

        if "tools_used" in parsed and any("save" in t.lower() for t in parsed["tools_used"]):
            should_save = True

        # Save if needed
        if should_save:
            print("\n Saving response using save_to_txt()...")
            result = save_to_txt(json.dumps(parsed, indent=2))
            print(result)
        else:
            print("\n No save tool requested — skipping file write.")

    except json.JSONDecodeError as e:
        print("JSON parse error:", e)

try:
    raw_string=parser.parse(raw_response.get("output"))
    structured_response=raw_string.model_dump().get("summary")
    print(type(structured_response))
    print(structured_response)
except Exception as e:
    print("Error parsing response:", e)
    print("Raw response:", raw_response)
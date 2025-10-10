import re
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.globals import set_debug
from tools import search_tool, wiki_tool, save_tool, save_to_txt

load_dotenv()
# load_dotenv(dotenv_path=".env.local")

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources:list[str]
    tools_used:list[str]

"""
# [IMPORTANT NOTE]: As of now, Gemini (Google Generative AI) does NOT support tool/function calling.
# Therefore, it only returns tool call intents in the output metadata, not actual executions.
# Based on which we call tools externally in this code.
# Only OpenAI and Anthropic models support function-based tool calling in LangChain.
# If you want to use tools, use ChatOpenAI or ChatAnthropic as your LLM.
"""
# llm=ChatOpenAI(model="gpt-4o", verbose=True)
# llm=ChatAnthropic(model="claude-sonnet-4-5-20250929", verbose=True)

llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7, verbose=False)

memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)

parser=PydanticOutputParser(pydantic_object=ResearchResponse)

prompt=ChatPromptTemplate.from_messages([
    ("system",
    """
    You are an helpful conversational research assistant. You will be given a topic to research. Use the tools at your disposal to gather information and provide a concise summary of the topic along with sources.
    Provide a summary, sources, and tools used in your response. Format your response as per the provided schema.
    Use the following format:\n{format_instructions}
    When user says to 'save', 'store', or 'write to file',
    output JSON ONLY in this format:

    {{
    "topic": "<topic name>",
    "summary": "<summary>",
    "sources": ["<source links>"],
    "tools_used": ["save_to_txt_file"]
    }}

    Otherwise, answer conversationally and naturally.
    Never say you cannot save — just output JSON when asked.
    """,
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools=[search_tool, wiki_tool, save_tool]
agent=create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor=AgentExecutor(agent=agent, memory=memory, tools=tools, verbose=False)

#Enable to see full traces of LLM calls and intermediate steps
set_debug(False)

print("\n+++ Chatbot is ready! Type 'exit' to quit. +++\n")

while True:
    query=input("You: ")
    if query.lower() in ['exit', 'quit']:
        print("Exiting...")
        break

    raw_response=agent_executor.invoke({"query":query})
    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, raw_response.get("output", ""), re.DOTALL)
    if not match:
        print("Bot: ", raw_response.get("output", ""))
    else:   
        json_str = match.group(1).strip()
        response_dict=json.loads(json_str)
        print("Bot: ", response_dict.get("summary"))

    #----Pseudo tool calling for Gemini model (since it doesn't support tool calling natively)----
    # Comment out below from #start to #end in case of OpenAI or Anthropic models
    # because they support tool calling natively and return final output directly.
    #start
    try:
        output_text = raw_response.get("output", "")

        # Extract JSON block (inside ```json ... ```)
        json_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_pattern, output_text, re.DOTALL)

        if not match:
            parsed={}
        else:
            json_str = match.group(1).strip()

            try:
                parsed = json.loads(json_str)

                # Check for explicit save intent
                should_save = False

                if "tools_used" in parsed and any("save" in t.lower() for t in parsed["tools_used"]):
                    should_save = True

                # Save if needed
                if should_save:
                    result = save_to_txt(json.dumps(parsed, indent=2))
                else:
                    pass

            except json.JSONDecodeError as e:
                print("JSON parse error:", e)
        # end
    except Exception as e:
        print("Error in pseudo tool calling:", e)

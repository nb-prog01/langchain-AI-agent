from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


def save_to_txt(data:str, filename:str="research_output.txt"):
    
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formated_text=f"----------Research Output----------\nTimestamp: {timestamp}\n{data}\n-------------------------------\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formated_text)
    return f"Data saved to {filename}"

save_tool=Tool(
    name="save_to_txt_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search=DuckDuckGoSearchRun()
search_tool=Tool(
    name="duckduckgo_search",
    func=search.run,
    description="Useful for when you need to look up current information on the web. Input should be a search query string.",
)

api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

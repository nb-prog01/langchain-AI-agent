# LangChain AI Agent Research Assistant

This project is an AI-powered research assistant that leverages [LangChain](https://github.com/langchain-ai/langchain), large language models, and web search tools to automate research tasks. It can search the web, summarize topics, and save structured research outputs.

## Features

- Uses Google Gemini, Anthropic Claude, or OpenAI models via LangChain.
- Integrates with DuckDuckGo and Wikipedia for up-to-date information.
- Outputs structured research summaries in JSON format.
- Optionally saves research results to a text file.
- Extensible tool system for adding new capabilities.

## Repository Structure

```
.env
.env.local
.gitignore
main.py
requirements.txt
research_output.txt
tools.py
__pycache__/
```

- **main.py**: Entry point. Runs the research agent, handles user input, and manages output parsing and saving.
- **tools.py**: Defines tools for web search, Wikipedia queries, and saving results.
- **requirements.txt**: Python dependencies.
- **research_output.txt**: Stores saved research outputs with timestamps.
- **.env / .env.local**: Environment variables for API keys and configuration (not committed).
- **.gitignore**: Ignores Python, environment, and IDE files.

## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   - Copy `.env` or `.env.local.example` to `.env.local` and add your API keys for OpenAI, Anthropic, or Google Gemini.

3. **Run the agent:**
   ```sh
   python main.py
   ```

4. **Enter a research query** when prompted.

## Output

- The agent prints a structured summary and sources.
- If the "save" tool is used, the output is appended to `research_output.txt` with a timestamp.

## Extending

- Add new tools in [`tools.py`](tools.py).
- Modify the prompt or agent logic in [`main.py`](main.py).

## License

This project is for research and educational purposes.
# AI Crew for Stock Analysis
## Introduction
This project is an example using the CrewAI framework to automate the process of analyzing a stock. CrewAI orchestrates autonomous AI agents, enabling them to collaborate and execute complex tasks efficiently. An additional step is added at the initial phase to enforce important data gathering for financial analysis. Areas such as technicals, fundamendals, Macro and market data (price) were considered. You can modify to whatever clients you think is important for financial analysis.s


## CrewAI Framework
CrewAI is designed to facilitate the collaboration of role-playing AI agents. In this example, these agents work together to give a complete stock analysis and investment recommendation

## Running the Script
It currently uses local ollama-model: gemma4:e4b. you can change it to stronger better such as GPT5 / Claude Opus 4.7, etc but beware of API costs.
Currently, the crewai ragtool has an in-built OPENAPI Key for chromadb vectorization. If you want to do it locally with a embedding model (Ollama) instead, need to remove ragtool and rebuild yourself.

- **Configure Environment**: Set up the environment variables for [SEC-API](https://sec-api.io) and [OpenAI](https://platform.openai.com/api-keys). You can use $env:SEC_API_KEY / OPENAI_APU_KEY = "xx" for quick testing.
- **API KEYS**: Create .env for your own designed clients for data ingestion layer. Modify the base_client to what api you are using. Update the config/setting.py for the Class settings with your own designed clients api key variables.
- **Install Dependencies**: Run `pip install -r requirements.txt`.

## Details & Explanation
- **Running the Script**: Execute `python main.py`` and input the company to be analyzed when prompted. The script will leverage the CrewAI framework to analyze the company and generate a detailed report.
- **Key Components**:
  - `./main.py`: Main script file.
  - `./stock_analysis_tasks.py`: Main file with the tasks prompts.
  - `./stock_analysis_agents.py`: Main file with the agents creation.
  - `./tools`: Contains tool classes used by the agents.



## Using Local Models with Ollama

- Instantiate Ollama Model: Create an instance of the Ollama model. You can specify the model and the base URL during instantiation. For example:

```python
from langchain.llms import Ollama
ollama_openhermes = Ollama(model="openhermes")
# Pass Ollama Model to Agents: When creating your agents within the CrewAI framework, you can pass the Ollama model as an argument to the Agent constructor. For instance:

def local_expert(self):
	return Agent(
      role='The Best Financial Analyst',
      goal="""Impress all customers with your financial data 
      and market trends analysis""",
      backstory="""The most seasoned financial analyst with 
      lots of expertise in stock market analysis and investment
      strategies that is working for a super important customer.""",
      verbose=True,
      llm=ollama_openhermes, # Ollama model passed here
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        CalculatorTools.calculate,
        SECTools.search_10q,
        SECTools.search_10k
      ]
    )
```
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from tools.calculator_tool import CalculatorTool
from tools.sec_tools import SEC10KTool, SEC10QTool

from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool

from dotenv import load_dotenv
load_dotenv()

llm = LLM(
    model="ollama/gemma4:e4b",
    base_url="http://localhost:11434"
)

@CrewBase
class StockAnalysisCrew:

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def financial_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_analyst"],
            verbose=True,
            llm=llm,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(),
                CalculatorTool(),
                SEC10QTool(stock_name="AMZN"),
                SEC10KTool(stock_name="AMZN"),
            ]
        )

    @agent
    def research_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["research_analyst"],
            verbose=True,
            llm=llm,
            tools=[
                ScrapeWebsiteTool(),
                SEC10QTool(stock_name="AMZN"),
                SEC10KTool(stock_name="AMZN"),
            ]
        )

    @agent
    def investment_advisor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["investment_advisor"],
            verbose=True,
            llm=llm,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(),
                CalculatorTool(),
            ]
        )

    @task
    def financial_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["financial_analysis"],
            agent=self.financial_analyst_agent(),
        )

    @task
    def research(self) -> Task:
        return Task(
            config=self.tasks_config["research"],
            agent=self.research_analyst_agent(),
        )

    @task
    def filings_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["filings_analysis"],
            agent=self.financial_analyst_agent(),
        )

    @task
    def recommend(self) -> Task:
        return Task(
            config=self.tasks_config["recommend"],
            agent=self.investment_advisor_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False
        )
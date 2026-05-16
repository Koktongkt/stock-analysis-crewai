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

    def __init__(self, inputs: dict = None):
        self.inputs = inputs or {}
        self.stock_name = self.inputs.get('company_stock', "")
        
    @agent
    def financial_analyst_agent(self) -> Agent:
        return Agent(
            role="Financial Analyst",
            goal="Analyze the financial statements, SEC filings, and market data to produce investment insights",
            backstory=(
                "You are a top-tier financial analyst with expertise in analyzing financial statements, SEC fillings, "
                "and market data to produce investment insights. You are highly detail-oriented and always support your "
                "findings with solid evidence."
            ),
            verbose=True,
            llm=llm,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(),
                CalculatorTool(),
                SEC10QTool(stock_name=self.stock_name),
                SEC10KTool(stock_name=self.stock_name),
            ]
        )

    @agent
    def research_analyst_agent(self) -> Agent:
        return Agent(
            role="Research Analyst",
            goal="Conduct comprehensive research on the company and its industry to provide actionable insights",
            backstory=(
                "You are a skilled research analyst with a deep understanding of the financial and market trends for all industries. "
                "You excel at gathering and synthesizing information from various sources to provide a complete picture "
                "of the company's position in the market. "
                "you're skilled in sifting through news, company announcements, and market sentiments."
            ),
            verbose=True,
            llm=llm,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(),
                SEC10QTool(stock_name=self.stock_name),
                SEC10KTool(stock_name=self.stock_name),
            ]
        )

    @agent
    def investment_advisor_agent(self) -> Agent:
        return Agent(
            role="Investment Advisor",
            goal="Based on the analysis and research, provide a clear recommendation on whether to buy, sell, or hold the stock.",
            backstory=(
                "You are a seasoned investment advisor with a strong track record of providing sound investment advice. "
                "You have a deep understanding of financial markets and are skilled at interpreting complex financial data to make informed recommendations. "
                "Your advice is always based on thorough analysis and research, and you are adept at communicating your recommendations clearly and persuasively."
            ),
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
            description=(
            "Conduct a thorough analysis of {company_stock}'s financial health and market performance. "
            "Examine key metrics like P/E ratio, EPS growth, revenue trends, and debt-to-equity ratio. "
            "Compare against industry peers and market trends."
        ),
            expected_output=(
            "A structured financial report covering strengths, weaknesses, "
            "valuation, and competitive positioning."
        ),    
            agent=self.financial_analyst_agent(),
        )

    @task
    def research(self) -> Task:
        return Task(
            description=(
            "Collect and summarize recent news, press releases, and market analysis "
            "related to {company_stock}. Focus on sentiment, analyst opinions, and upcoming events."
        ),
            expected_output=(
            "A structured news summary highlighting sentiment shifts and catalysts affecting the stock."
        ),    
            agent=self.research_analyst_agent(),
        )

    @task
    def filings_analysis(self) -> Task:
        return Task(
            description=(
            "Analyze latest 10-Q and 10-K filings for {company_stock}. "
            "Focus on MD&A, financial statements, insider activity, and risks."
        ),
        expected_output=(
            "A detailed SEC filings analysis highlighting risks, strengths, and anomalies."
        ),
            agent=self.financial_analyst_agent(),
        )

    @task
    def recommend(self) -> Task:
        return Task(
            description=(
            "Synthesize financial analysis, research, and filings into a final investment recommendation "
            "for {company_stock}."
        ),
        expected_output=(
            "A well-structured investment report with clear buy/hold/sell stance and justification."
        ),
            agent=self.investment_advisor_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks= [
                self.financial_analysis(),
                self.research(),
                self.filings_analysis(),
                self.recommend()              
            ]
            process=Process.sequential,
            verbose=True,
            memory=False
        )
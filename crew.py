from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from tools.calculator_tool import CalculatorTool
from tools.sec_tools import SEC10KTool, SEC10QTool

from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool
from data.ingestion_pipeline import IngestionPipeline


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

        # Initialize clients
        self.pipeline = IngestionPipeline(symbol=self.stock_name)
        self.research_packet = self.pipeline.build_research_packet()

        # Initialize expensive tools ONCE
        self.sec10q_tool = SEC10QTool(stock_name=self.stock_name)
        self.sec10k_tool = SEC10KTool(stock_name=self.stock_name)

        # Shared lightweight tools
        self.scrape_tool = ScrapeWebsiteTool()
        self.search_tool = WebsiteSearchTool()
        self.calculator_tool = CalculatorTool()

        
    @agent
    def financial_analyst_agent(self) -> Agent:
        return Agent(
            role="Financial Analyst",
            goal=(
                f"Analyze {self.stock_name}'s financial statements, SEC filings, and market data to produce investment insights"
            ),
            backstory=(
                "You are a veteran Wall Street financial analyst with expertise in "
                "fundamental analysis, SEC filing interpretation, valuation modeling, "
                "and identifying financial risks and growth drivers. "
                "You provide evidence-backed analysis using quantitative and qualitative data."
            ),
            verbose=True,
            llm=llm,
            tools=[
                self.scrape_tool,
                self.search_tool,
                self.calculator_tool,
                self.sec10q_tool,
                self.sec10k_tool,
            ]
        )

    @agent
    def research_analyst_agent(self) -> Agent:
        return Agent(
            role="Equity Research Analyst",

            goal=(
                f"Research {self.stock_name}'s market environment, "
                "industry trends, news sentiment, competitive positioning, "
                "and macroeconomic catalysts."
            ),

            backstory=(
                "You are an elite equity research analyst specializing in "
                "market intelligence, industry analysis, earnings sentiment, "
                "and identifying catalysts affecting stock performance. "
                "You synthesize information from filings, news, and industry developments."
            ),

            verbose=True,
            llm=llm,

            tools=[
                self.scrape_tool,
                self.search_tool,
                self.sec10q_tool,
                self.sec10k_tool,
            ]
        )

    @agent
    def investment_advisor_agent(self) -> Agent:
        return Agent(
            role="Chief Investment Strategist",

            goal=(
                f"Synthesize all research and analysis on {self.stock_name} "
                "to provide a professional investment recommendation and risk assessment."
            ),

            backstory=(
                "You are a seasoned investment strategist managing institutional portfolios. "
                "You specialize in converting complex financial analysis into actionable "
                "investment recommendations with clear risk/reward tradeoffs."
            ),

            verbose=True,
            llm=llm,

            tools=[
                self.scrape_tool,
                self.search_tool,
                self.calculator_tool,
            ]
        )

    @task
    def financial_analysis(self) -> Task:
        return Task(
            description=(
            "Conduct a thorough analysis of {company_stock}'s financial health and market performance. "
            "Examine key metrics like P/E ratio, EPS growth, revenue trends, and debt-to-equity ratio. "
            "Compare against industry peers and market trends."
            "Use information up till {today}."
            "You are also given the following research packet to analyze:\n\n"
            "FUNDAMENTALS:\n{fundamentals}\n\n"
            "TECHNICALS:\n{technicals}\n\n"
            "MARKET DATA:\n{market_data}\n\n"
        ),
            expected_output=(
            "A structured financial report covering its fundamental and technical strengths, weaknesses, "
            "valuations, and competitive positioning."
        ),    
            agent=self.financial_analyst_agent(),
        )

    @task
    def research(self) -> Task:
        return Task(
            description=(
            "Collect and summarize recent news, press releases, and market analysis "
            "related to {company_stock}, up to {today}. Focus on sentiment, analyst opinions, and upcoming events."
            "You are also given the following research packet to analyze:\n\n"
            "MACRO:\n{macro}\n\n"
            "Focus on catalysts, sentiment shifts, and macro risks."
        ),
            expected_output=(
            "A structured news summary highlighting sentiment and macro shifts and catalysts affecting the stock."
        ),    
            agent=self.research_analyst_agent(),
        )

    @task
    def filings_analysis(self) -> Task:
        return Task(
            description=(
            "Analyze latest 10-Q and 10-K filings for {company_stock} until {today} "
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
            "for {company_stock}, as of {today}\n."
            "Available information includes:\n\n"
            "FUNDAMENTALS:\n{fundamentals}\n\n"
            "TECHNICALS:\n{technicals}\n\n"
            "MARKET DATA:\n{market_data}\n\n"
            "MACRO:\n{macro}\n\n"
            "Generate final investment recommendation (Buy/Hold/Sell) with a clear rationale and risk assessment."
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
            ],
            process=Process.sequential,
            verbose=True,
            memory=False
        )
    
    def run(self):
        return self.crew().kickoff(
            inputs={
                **self.inputs,

                "fundamentals": self.research_packet["fundamentals"],
                "technicals": self.research_packet["technicals"],
                "macro": self.research_packet["macro"],
                "market_data": self.research_packet.get("market_data"),
            }
        )
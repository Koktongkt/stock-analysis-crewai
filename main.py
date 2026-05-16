import sys
from crew import StockAnalysisCrew

def run():
    print("\n--- Stock Analysis Configuration ---")
    # Prompt the user dynamically in the terminal
    company_stock = input("Enter the stock ticker symbol (e.g., AAPL, AMZN, MSFT): ").strip().upper()
    #query = input("What specific analysis objective do you have? (e.g., Last year's revenue): ").strip()
    
    if not company_stock:
        print("❌ Error: Stock ticker cannot be empty. Please provide a company ticker symbol.")
        return

    inputs = {
        'company_stock': company_stock,
    }
    
    print(f"\n🚀 Kickstarting crew to analyze {company_stock}...\n")
    return StockAnalysisCrew(inputs=inputs).crew().kickoff(inputs=inputs)

def train():
    """
    Train the crew for a given number of iterations.
    """
    print("\n--- Crew Training Configuration ---")
    company_stock = input("Enter the stock ticker for training data: ").strip().upper()
    #query = input("Enter the training query objective: ").strip()

    inputs = {
        'company_stock': company_stock,
    }
    try:
        iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        print(f"\n🏋️ Training the crew for {iterations} iterations on {company_stock}...")
        StockAnalysisCrew(inputs=inputs).crew().train(n_iterations=iterations, inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
    
if __name__ == "__main__":
    print("## Welcome to Stock Analysis Crew")
    print('-------------------------------')
    
    # Check if user wants to train or run standard execution
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        train() # type your command like: python main.py 10 (to train for 10 iterations)
    else:
        result = run() # type python main.py for a single run
        if result:
            print("\n\n########################")
            print("## Here is the Report")
            print("########################\n")
            print(result)
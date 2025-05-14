import logging
from datetime import datetime, timedelta
import yfinance as yf
from dateutil import parser

from model_openai.openai_llm import OpenAILLM
from reason_and_act.react_agent import ReactAgent
from tool_use.llm_tool import convert_to_llm_tool as llm_tool


def calculate_price_growth_func(start_price: str, end_price: str) -> str:
    """
    Calculate the growth rate between two prices.

    Parameters:
    start_price (float): The starting price.
    end_price (float): The ending price.

    Returns:
    str: Message containing the growth rate between the two prices
    """
    # Sometimes it is better to handle parsing in the function so you can return a meaningful error
    try:
        start_price_num = float(start_price)
        end_price_num = float(end_price)
    except Exception:
        raise ValueError(
            "Both start_price and end_price must be numbers. Call calculate_price_growth_func again, once you have the actual prices."
        )
    if start_price_num == 0:
        start_price_num = 1 / float("inf")
    ret = (end_price_num - start_price_num) / start_price_num
    # Sometimes more humanised responses result in more accurate answers
    return (
        f"The growth rate between {start_price_num} and {end_price_num} is {ret:.4f}."
    )


def get_spot_price_func(ticker_symbol: str, date: str) -> str:
    """
    Get the market close price for a given ticker symbol and date.

    Parameters:
    ticker_symbol (str): the ticker symbol of the asset
    date (datetime.date): the date for which the price is required

    Returns:
    str: Message containing the market close price
    """
    # Sometimes it is better to handle parsing in the function
    date_converted = parser.parse(date)
    ret = get_spot_price(ticker_symbol, date_converted)
    # Sometimes more humanised responses result in more accurate answers
    return f"The market close price for {ticker_symbol} on {date_converted.strftime('%Y-%m-%d')} is {ret:.4f}."


def get_spot_price(ticker_symbol: str, date: datetime.date) -> float:
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(interval="1d", start=date, end=date + timedelta(days=1))
    ret = hist["Close"].iloc[-1]
    return ret


def run_react_agent():
    """
    Run the react agent to showcase agent's ability to plan and use different tools.

    Parameters:
    None

    Returns:
    None
    """
    llm = OpenAILLM(
        "meta-llama/llama-3.2-3b-instruct/fp-16", "https://api.inference.net/v1",
        # "meta-llama/llama-3.3-70b-instruct/fp-16", "https://api.inference.net/v1"
    )
    llm_tools = [
        llm_tool(get_spot_price_func),
        llm_tool(calculate_price_growth_func),
    ]
    agent = ReactAgent(llm, llm_tools)

    last_wednesday = datetime.now() - timedelta(
        days=(datetime.now().weekday() + 4) % 7 + 7
    )
    last_wednesday_before = datetime.now() - timedelta(
        days=(datetime.now().weekday() + 4) % 7 + 14
    )
    user_prompt = f"Was the price growth rate between {last_wednesday_before.date()} and {last_wednesday.date()} higher for NVDA or PLTR?"
    # user_prompt = f"Was NVDA or PLTR closing price higher on {last_wednesday.date()}?"

    response = agent.generate(user_prompt)
    logging.info("The response is:")
    logging.info(response)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_react_agent()

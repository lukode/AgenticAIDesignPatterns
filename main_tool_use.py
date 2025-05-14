import logging
from datetime import datetime, timedelta
import yfinance as yf
import json
from dateutil import parser

from model_openai.openai_llm import OpenAILLM
from tool_use.llm_tool import convert_to_llm_tool as llm_tool
from tool_use.tool_use_agent import (
    ToolUseAgent,
)


def get_current_temperature_func(location: str, unit: str):
    """
    Get the temperature for a given location

    Parameters:
    location (str): The location, for example 'London' or 'New York'
    unit (str): The temperature unit. Either 'celsius' or 'fahrenheit'.
    """
    return (
        json.dumps({"temperature": 25, "unit": unit})
        if location == "New York"
        else json.dumps({"temperature": 15, "unit": unit})
    )


def get_spot_price_func(ticker_symbol: str, date: str) -> float:
    """
    Get the market close price for a given ticker symbol and date.

    Parameters:
    ticker_symbol (str): the ticker symbol of the asset
    date (datetime.date): the date for which the price is required

    Returns:
    float: the market close price
    """
    # Sometimes it is better to handle parsing in the function
    date_converted = parser.parse(date)
    return get_spot_price(ticker_symbol, date_converted)


def get_spot_price(ticker_symbol: str, date: datetime.date) -> float:
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(interval="1d", start=date, end=date + timedelta(days=1))
    ret = hist["Close"].iloc[-1]
    return ret


def run_tool_agent():
    """
    Run the tool use agent to showcase agent's the tool use capabilities.

    Parameters:
    None

    Returns:
    None
    """
    llm = OpenAILLM(
        "meta-llama/llama-3.2-3b-instruct/fp-16", "https://api.inference.net/v1"
        # "meta-llama/llama-3.3-70b-instruct/fp-16", "https://api.inference.net/v1"
    )
    llm_tools = [
        llm_tool(get_current_temperature_func),
        llm_tool(get_spot_price_func),
    ]
    agent = ToolUseAgent(llm, llm_tools)

    last_wednesday = datetime.now() - timedelta(
        days=(datetime.now().weekday() + 4) % 7 + 7
    )
    user_prompt = f"What was the closing price of MSFT on {last_wednesday.date()}?"
    # user_prompt = "What's the temperature in London?"

    response = agent.generate(user_prompt)
    logging.info("The response is:")
    logging.info(response)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tool_agent()

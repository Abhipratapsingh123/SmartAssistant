import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
from langchain_core.tools import tool
import requests


# ------------------ Load ENV ------------------

# st.secrets are correctly configured in your Streamlit environment
api_key_weather = st.secrets["API_KEY_WEATHER"]
api_key_currency_converter = st.secrets["API_KEY_CURRENCY_CONVERTER"]
api_holiday_key = st.secrets["API_HOLIDAY_KEY"]



# ------------------ Tools ------------------


# tool to fetch real date and time
@tool
def current_datetime() -> str:
    """Returns the current system date and time."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y - %I:%M %p")



# tool to make real time web search
@tool
def search_tool(query: str) -> str:
    """
    Perform a real-time web search using DuckDuckGo.
    Use this tool to fetch live information such as today's date, current events, or news.
    
    Args:
        query (str): The search query or topic to look up.
    
    Returns:
        str: The summarized search result text.
    """
    search_instance = DuckDuckGoSearchRun()
    result = search_instance.invoke(query)
    return result



# tool to fetch weather of any location within india and can forecast upto 3 days
@tool
def weather_forecast(location: str = 'India', days: int = 3) -> dict:
    """Fetches the weather forecast for a given location in India (default: 3-day forecast).User must enter the city name for this tool."""
    
    location_query = f"{location},IN"
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key_weather}&q={location_query}&days={days}&aqi=yes&alerts=yes"
    response = requests.get(url)
    data = response.json()
    if "error" in data:
        return {"error": data["error"]["message"]}
    
    # Pruning the forecast data for a cleaner tool output
    forecast = {
        "location": data["location"]["name"],
        "current_temp_C": data["current"]["temp_c"],
        "condition": data["current"]["condition"]["text"],
        "forecast_summary": [{
            "date": day["date"],
            "max_temp_C": day["day"]["maxtemp_c"],
            "min_temp_C": day["day"]["mintemp_c"],
            "condition": day["day"]["condition"]["text"],
            "rain_chance": day["day"]["daily_chance_of_rain"]
        } for day in data["forecast"]["forecastday"]]
    }
    return forecast

# tool to fetch live currency rates and convert to any other currency
@tool
def convert_currency_amount(amount: float, from_currency: str, to_currency: str) -> float:
    """
    Converts a specific amount from one currency to another using a real-time exchange rate.
    
    Args:
        amount: The value to be converted (e.g., 1000).
        
        from_currency: The 3-letter code of the source currency (e.g., USD, EUR, INR), (if user provide country name instead of code then first convert it to suitable country code.)
        
        to_currency: The 3-letter code of the target currency (e.g., USD, EUR, INR), (if user provide country name instead of code then first convert it to suitable country code.)
        
    Returns: The converted amount as a float, or if this tool doesnot work then search real currency rate using other tool.
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    
    url = f"https://v6.exchangerate-api.com/v6/{api_key_currency_converter}/pair/{from_currency}/{to_currency}"
    response = requests.get(url)
    data = response.json()
    
    if data.get("result") == "error":
        return f"Error fetching rate: {data.get('error-type', 'Unknown error')}"
        
    conversion_rate = data.get("conversion_rate")
    
    if conversion_rate is None:
        return f"Could not find conversion rate for {from_currency} to {to_currency}."
        
    converted_amount = amount * conversion_rate
    return round(converted_amount, 2)


# tool to get data about holidays on specific date in India

@tool
def get_holiday(date: str, country: str = "IN") -> dict:
    """Fetches holiday information for a given date and country.
    - date must be in format YYYY-MM-DD.
    - country must be a 2-letter ISO code (e.g., "IN" for India).
    """
    year, month, day = date.split("-")
    url = f"https://holidays.abstractapi.com/v1/?api_key={api_holiday_key}&country={country}&year={year}&month={month}&day={day}"
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code}"}
    data = response.json()
    if not data:
        return {"message": f"No holiday on {date} in {country}"}
    return {
        "date": data[0].get("date"),
        "name": data[0].get("name"),
        "type": data[0].get("type"),
        "location": data[0].get("location"),
    }


# --- Unique Super Tool (Demonstrates complex reasoning) ---
@tool
def travel_quick_planner(destination_city: str, date: str) -> str:
    """
    Provides a quick travel summary for a given city and date, combining weather and holiday checks.
    This tool should be used when the user asks for a high-level plan or overview.
    
    Args:
        destination_city: The name of the city (e.g., 'Agra', 'Bangalore').
        date: The date for the check in YYYY-MM-DD format (e.g., '2025-12-25').
        
    Returns: A comprehensive summary detailing the weather and any observed holiday.
    """
    return f"To generate a quick travel summary for {destination_city} on {date}, the agent will now proceed to use the `weather_forecast`(to forecast upto3 days only),`Search tool (if needed),`budget_context`and `get_holiday` tools in sequence."


@tool
def budget_context_search(city_name: str, item_category: str = 'daily budget per person') -> str:
    """
    Fetches real-time cost-of-living or travel budget insights in indian rupees for a given city.
    Use this to help users understand the purchasing power or estimated expenses.

    Args:
        city_name (str): The city to analyze (e.g., 'Agra', 'Mumbai').
        item_category (str): The type of expense to check 
                             (e.g., 'daily budget per person', 'local meal price', 'average hotel rate').

    Returns:
        str: A concise summary of cost-related information retrieved from live web data.
    """
    query = f"Average cost of {item_category} in {city_name} in Indian rupees"
    search_result = search_tool(query)
    return f"Cost context for {city_name} ({item_category}): {search_result}"
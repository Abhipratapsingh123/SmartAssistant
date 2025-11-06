import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain import hub
import requests
import io
import sys
from langchain.callbacks import StreamlitCallbackHandler 

# ------------------ Load ENV ------------------
# Assume st.secrets are correctly configured in your Streamlit environment
api_key_weather = st.secrets["API_KEY_WEATHER"]
api_key_currency_converter = st.secrets["API_KEY_CURRENCY_CONVERTER"]
api_holiday_key = st.secrets["API_HOLIDAY_KEY"]
api_key = st.secrets["GOOGLE_API_KEY"]

# ------------------ Tools ------------------

@tool
def current_datetime() -> str:
    """Returns the current system date and time."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y - %I:%M %p")



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

# --- Currency Tool ---
@tool
def convert_currency_amount(amount: float, from_currency: str, to_currency: str) -> float:
    """
    Converts a specific amount from one currency to another using a real-time exchange rate.
    
    Args:
        amount: The value to be converted (e.g., 1000).
        from_currency: The 3-letter code of the source currency (e.g., USD, EUR, INR).
        to_currency: The 3-letter code of the target currency (e.g., USD, EUR, INR).
        
    Returns: The converted amount as a float, or an error message.
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


# -----------------------------

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
    return f"To generate a quick travel summary for {destination_city} on {date}, the agent will now proceed to use the `weather_forecast`.`budget_context`and `get_holiday` tools in sequence."


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



# Update the tools list with the new/modified tools
#  RemovED 'get_conversion_factor' and 'convert' and add 'convert_currency_amount' and 'travel_quick_planner'

tools = [current_datetime,search_tool, weather_forecast, convert_currency_amount, get_holiday,budget_context_search, travel_quick_planner]

# ------------------ Agent Setup ------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Think Mate", layout="wide")
st.title("🤖 Think Mate - Advanced AI Agent")

st.markdown("### ✈️ **Travel Planning** |💱**Unified Currency** | ☁️ Weather | 🎉 Holidays | 🔎Search")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # The agent's thinking process will now be shown via the callback handler, 
    # but we'll keep the StreamlitCallbackHandler active by default for a smoother UI.
    st.markdown("The agent's thinking process (Tool Calls) will now appear in the chat output.")
    
    # Session state for chat history initialization
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful and  advanced AI agent who assists users with complex queries by intelligently utilizing your tools. If anyone asks you who made you, say `I am an AI agent made by Abhi Pratap Singh`."),
        ]
    
    # Download chat history
    if st.button("Download Chat History"):
        chat_text = "\n".join([f"{msg.type}: {msg.content}" for msg in st.session_state.chat_history])
        st.download_button("Download", chat_text, file_name="chat_history.txt")

# Display previous chat
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg.content)

# User input
if user_input := st.chat_input("Ask me something..."):
    # Add user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)

    # Use the StreamlitCallbackHandler for better log/tool call visualization
    st_callback = StreamlitCallbackHandler(st.container())
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            # Pass the callback handler to the invoke method
            result = agent_executor.invoke(
                {"input": user_input, "chat_history": st.session_state.chat_history},
                {"callbacks": [st_callback]} # Pass the handler here
            )
            
            ai_response = result.get("output", str(result))
            
            # Display the final AI response *after* the thinking is done
            st.write(ai_response)

    # Save AI response
    st.session_state.chat_history.append(AIMessage(content=ai_response))

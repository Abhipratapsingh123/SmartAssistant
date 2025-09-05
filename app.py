import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain import hub
import requests
import io
import sys

# ------------------ Load ENV ------------------
api_key_weather = st.secrets["API_KEY_WEATHER"]
api_key_currency_converter = st.secrets["API_KEY_CURRENCY_CONVERTER"]
api_holiday_key = st.secrets["API_HOLIDAY_KEY"]
api_key = st.secrets["GOOGLE_API_KEY"]

# ------------------ Tools ------------------
search_tool = DuckDuckGoSearchRun()

@tool
def weather_forecast(location: str = 'India', days: int = 7) -> dict:
    """Fetches the weather forecast for a given location in India (default: 7-day forecast)."""
    location_query = f"{location},IN"
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key_weather}&q={location_query}&days={days}&aqi=yes&alerts=yes"
    response = requests.get(url)
    data = response.json()
    if "error" in data:
        return {"error": data["error"]["message"]}
    forecast = {
        "location": data["location"]["name"],
        "region": data["location"]["region"],
        "country": data["location"]["country"],
        "current_temp_C": data["current"]["temp_c"],
        "condition": data["current"]["condition"]["text"],
        "forecast_days": []
    }
    for day in data["forecast"]["forecastday"]:
        forecast["forecast_days"].append({
            "date": day["date"],
            "max_temp_C": day["day"]["maxtemp_c"],
            "min_temp_C": day["day"]["mintemp_c"],
            "condition": day["day"]["condition"]["text"],
            "daily_chance_of_rain": day["day"]["daily_chance_of_rain"]
        })
    return forecast

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetch currency conversion rate."""
    url = f"https://v6.exchangerate-api.com/v6/{api_key_currency_converter}/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    data = response.json()
    return data.get("conversion_rate")

@tool
def convert(base_currency_value: int, conversion_rate: float) -> float:
    """Convert currency value using conversion rate."""
    return base_currency_value * conversion_rate

@tool
def get_holiday(date: str, country: str = "IN") -> dict:
    """Fetch holiday info for a given date in YYYY-MM-DD format."""
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

tools = [search_tool, weather_forecast, get_conversion_factor, convert, get_holiday]

# ------------------ Agent Setup ------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Think Mate", layout="wide")
st.title("🤖 Think Mate - Your Smart AI Assistant")

st.markdown("### ☁️ Weather | 🎉 Holidays | 💱 Currency | 🔎 Search")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    show_logs = st.checkbox("Show agent thinking process")
    # Download chat history
    if st.button("Download Chat History"):
        chat_text = "\n".join([f"{msg.type}: {msg.content}" for msg in st.session_state.chat_history])
        st.download_button("Download", chat_text, file_name="chat_history.txt")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant. If anyone asks who made you, say `I am an agent made by Abhi Pratap Singh`."),
    ]

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

    # Capture agent logs
    log_capture = io.StringIO()
    sys.stdout = log_capture

    # Call agent
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            result = agent_executor.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
            ai_response = result.get("output", str(result))
            st.write(ai_response)

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Save AI response
    st.session_state.chat_history.append(AIMessage(content=ai_response))

    # Show captured logs if enabled
    if show_logs:
        with st.expander("🔍 Agent Thinking Process"):
            st.text(log_capture.getvalue())



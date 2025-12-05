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
    """Returns the current system date and time.
       Use this tool when the user asks for today's date, time, or current timestamp.
    """
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
def weather_forecast(location: str = 'India', days: int = 14) -> dict:
    """Fetches the weather forecast for a given location in India (default: 14-day forecast).User must enter the city name for this tool."""
    
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



@tool
def safety_risk_radar(city: str) -> dict:
    """
    Provides a consolidated safety and risk assessment for any Indian city.
    Uses live search + weather alerts + regional risk checks.
    
    Key Outputs:
    - Crime trend summary
    - Flood/landslide warnings
    """

    # -------------------- 1. Crime Trend Summary --------------------
    crime_query = f"latest crime trends and safety situation in {city} India 2025"
    crime_info = search_tool(crime_query)

    # -------------------- 2. Flood / Landslide Risk (Uttarakhand-specific) --------------------
    terrain_query = f"flood and landslide risk level in {city} India"
    terrain_info = search_tool(terrain_query)

    # -------------------- Consolidated Risk Output --------------------
    output = {
        "city": city,
        "crime_trends": crime_info,
        "terrain_risk": terrain_info
    }

    return output





## new tool

@tool
def foursquare_places(city: str, query: str = "restaurants", radius_m: int = 2000, limit: int = 10) -> dict:
    """
    Search Foursquare for places in a given city using a simple category keyword.

    Notes:
    - `city` must be a real city name (e.g., Delhi, Mumbai).
    - `query` must be a single word category like: restaurants, cafes, hotels, bars.
    - Returns places ONLY (no photos).

    Args:
        city: City name.
        query: One-word place type.
        radius_m: Search radius in meters.
        limit: Number of results.

    Returns:
        Dictionary with: city coords, result count, and list of places.
    """

    SERVICE_KEY = st.secrets["SERVICE_KEY"]
    API_VERSION = "2025-06-17"

    # Step 1: Convert city name → coordinates
    geo = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": city, "format": "json", "limit": 1},
        headers={"User-Agent": "ThinkMate-Agent"}
    ).json()

    if not geo:
        return {"error": f"City '{city}' not found"}

    lat, lon = geo[0]["lat"], geo[0]["lon"]

    # Step 2: Foursquare search (NO photo fields)
    url = "https://places-api.foursquare.com/places/search"
    headers = {
        "Authorization": f"Bearer {SERVICE_KEY}",
        "X-Places-Api-Version": API_VERSION,
        "Accept": "application/json"
    }

    params = {
        "ll": f"{lat},{lon}",
        "query": query,
        "radius": radius_m,
        "limit": limit
    }

    res = requests.get(url, headers=headers, params=params).json()
    places = res.get("results", [])

    results = []
    for place in places:
        results.append({
            "name": place.get("name"),
            "address": place.get("location", {}).get("formatted_address"),
            "distance_m": place.get("distance"),
            "categories": [c.get("name") for c in place.get("categories", [])],
            "fsq_place_id": place.get("fsq_place_id")
        })

    # Save for follow-up queries (get_place_id_by_index, etc.)
    try:
        import streamlit as st
        st.session_state.last_places = results
    except:
        pass

    return {
        "city": city,
        "coords": {"lat": lat, "lon": lon},
        "results_count": len(results),
        "results": results
    }


    

## photo tool

@tool
def city_photos(city: str, count: int = 5) -> list:
    """
    Fetches high-quality city images using Unsplash.
    Use this tool to display destination visuals during trip planning.
    
    When a user asks for photos of any city, destination, or place,
    you MUST call the `city_photos` tool immediately.

    Args:
        city: City name (e.g., Delhi, Mumbai).
        count: Number of images to return.

    Returns:
        A list of image URLs (regular size).
    """
    key = st.secrets["UNSPLASH_ACCESS_KEY"]
    url = "https://api.unsplash.com/search/photos"
    params = {"query": city, "per_page": count, "client_id": key}

    res = requests.get(url, params=params).json()
    photos = [p["urls"]["regular"] for p in res.get("results", [])]

    return {"type": "photos", "urls": photos}





# --- Unique Super Tool (Demonstrates complex reasoning) ---
@tool
def full_trip_planner(city: str, start_date: str, end_date: str, budget: int = 50000) -> str:
    """
    Use this tool when the user requests a COMPLETE trip plan that includes:
    - weather
    - holidays
    - safety
    - budget
    - places to visit
    - photos

    This tool does NOT generate the content itself.
    Instead, it instructs the agent to sequentially use:
      1. weather_forecast tool
      2. get_holiday tool
      3. safety_risk_radar tool
      4. budget_context_search tool
      5. foursquare_places tool
      6. city_photos tool

    The agent MUST perform these calls after this tool is triggered.
    """
    return f"""
    Start full trip planning for:
    City: {city}
    Dates: {start_date} to {end_date}
    Budget: {budget}

    Now perform these steps in order:
    1. Call weather_forecast tool(city)
    2. Call get_holiday tool(start_date)
    3. Call safety_risk_radar tool(city)
    4. Call budget_context_search tool(city)
    5. Call foursquare_places tool(city, 'tourist')
    6. Call city_photos tool(city)

    Then summarize all tool results beautifully.
    """




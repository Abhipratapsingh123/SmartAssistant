import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain import hub
import io
import sys
from langchain.callbacks import StreamlitCallbackHandler 
from tools_module import (
    current_datetime, search_tool, weather_forecast,
    get_holiday, convert_currency_amount,
    budget_context_search, travel_quick_planner
)

# ------------------ Load ENV ------------------
api_key = st.secrets["GOOGLE_API_KEY"]


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
st.title("🤖 Think Mate - Trip Panning-AI Agent")

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
            SystemMessage(content="You are a helpful and  advanced AI agent who assists users with  queries by intelligently utilizing your tools. If anyone asks you who made you, say `I am an AI agent made by Abhi Pratap Singh`."),
        ]
    
    # Download chat history
    if st.button("Download Chat History"):
        chat_text = "\n".join([f"{msg.type}: {msg.content}" for msg in st.session_state.chat_history])
        st.download_button("Click here", chat_text, file_name="chat_history.txt")

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

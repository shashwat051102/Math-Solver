import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
import os
load_dotenv()

# Set up the Streamlit app
st.set_page_config(page_title="Math Solver", page_icon=":guardsman:", layout="wide")
st.title("Text to math problem solver")


groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="Llama-3.3-70b-Specdec", groq_api_key=groq_api_key)


# INitializing the tools

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = "Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find the various info on the topics mentioned"
)

# Initialize the math tool

math_chain = LLMMathChain.from_llm(llm=llm)

calculator = Tool(
    name = "Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expressions need to be provided"
)

prompt = """
You are a agent task for solving users mathematical question. Logically arrive at the solution and provide explanation 
and display it point wise for the question below 
Question:{question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all the tools into chain

chain = LLMChain(llm=llm,prompt = prompt_template)

# To add the reasoning inside this

reasoning_tool = Tool(
    name = "Reasoning",
    func = chain.run,
    description = "A tool for answering logic based and reasoning questions"
)

# initialize the agents

# Note = verbose is the reason why thought chain is showed in the streamlit app
assistant_agent = initialize_agent(
    tools = [wikipedia_tool, calculator, reasoning_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hello! I am your math assistant. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# function to generate the response

# def generate_response(question):
#     response = assistant_agent.invoke({'invoke':question})
#     return response
    
# Lets start the interaction

question = st.text_area("Enter the question:")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate Response...."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages,callbacks = [st_cb])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write('###Response')
            st.success(response)
    else:
        st.error("Please enter a question")
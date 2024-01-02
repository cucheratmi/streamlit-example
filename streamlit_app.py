import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from llama_index import download_loader
import os

st.set_page_config(page_title="SFPT pharmaCovid", layout="centered",initial_sidebar_state="auto", menu_items=None)

st.title("Médicaments & COVID")
st.info("les pages [pharmaCovid](https://sfpt-fr.org/covid19)")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Posez moi une question sur médicaments et COVID!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    urls = [
        "https://sfpt-fr.org/covid19-foire-aux-questions/1373-168-l-ivermectine-antiparasitaire-est-elle-efficace-pour-pr%C3%A9venir-ou-traiter-une-infection-%C3%A0-la-covid-19",
        "https://sfpt-fr.org/faq-vaccins",
        "https://sfpt-fr.org/covid19-foire-aux-questions/1094-la-chloroquine-ou-l-hydroxychloroquine-sont-elles-efficaces-pour-prevenir-ou-traiter-l-infection-par-coronavirus",
        "https://sfpt-fr.org/covid19-foire-aux-questions/1579-180-que-peut-on-attendre-des-antiviraux-dans-le-traitement-de-la-covid-19,-quels-sont-les-m%C3%A9canismes-d%E2%80%99action",
        "https://sfpt-fr.org/covid19-foire-aux-questions/1602-181-qu-est-ce-que-le-paxlovid%C2%AE-est-il-efficace-pour-traiter-la-covid-19",
        "https://sfpt-fr.org/covid19-foire-aux-questions/1326-164-la-prise-de-dexamethasone-am%C3%A9liore-t-elle-l%E2%80%99%C3%A9tat-clinique-des-patients-atteints-de-covid-19-hospitalis%C3%A9s-en-r%C3%A9animation-avec-atteinte-respiratoire-grave",
        "https://sfpt-fr.org/covid19-foire-aux-questions/1451-178-qu-en-est-il-de-l%E2%80%99efficacit%C3%A9-de-la-fluvoxamine-et-des-autres-anti-d%C3%A9presseurs-pour-le-traitement-de-la-covid-19%20",
        "https://sfpt-fr.org/covid19-foire-aux-questions/1302-160-les-interf%C3%A9rons-qu-est-ce-que-c-est-sont-ils-efficaces-pour-pr%C3%A9venir-ou-traiter-le-covid-19",
        "https://sfpt-fr.org/covid19-foire-aux-questions/1435-174-quelle-est-l%E2%80%99efficacit%C3%A9-des-anticorps-iv-th%C3%A9rapeutiques-contre-la-covid-19",
        "https://sfpt-fr.org/covid19-foire-aux-questions/1385-171-la-vitamine-d-est-elle-efficace-pour-pr%C3%A9venir-ou-traiter-la-covid-19",
        "https://sfpt-fr.org/covid19-foire-aux-questions/1107-l-azithromycine-est-elle-efficace-pour-prevenir-ou-traiter-l-infection-par-coronavirus",
        "https://sfpt-fr.org/covid19-foire-aux-questions/1275-154-quelle-est-l%E2%80%99efficacit%C3%A9-des-anticorps-dirig%C3%A9s-contre-les-m%C3%A9diateurs-de-l%E2%80%99inflammation-dans-la-covid-19-tocilizumab,-sarilumab,-eculizumab,-anakinra,-lenzilumab%E2%80%A6",
        "https://sfpt-fr.org/covid19-foire-aux-questions/1092-chloroquine-hydroxychloroquine-nivaquine-plaquenil-c-est-quoi",
    ]
    with st.spinner(text="Lecture des pages pharmacovid de la SFPT ! Un peu de patience SVP."):

        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo", temperature=0.5,
                system_prompt="You are an expert on pharmacology and covid and your job is to answer technical questions. Assume that all questions are related to the covid. Keep your answers technical and based on facts – do not hallucinate features."
                )
        )
        SimpleWebPageReader = download_loader("SimpleWebPageReader", custom_path="./tempo")
        loader = SimpleWebPageReader()
        documents = loader.load_data(urls=urls)
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Votre question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("En reflexion (profonde)..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history

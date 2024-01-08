import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from llama_index import download_loader
import os

st.set_page_config(page_title="SFPT pharmaCovid", layout="centered",initial_sidebar_state="auto", menu_items=None)

st.subheader("Société Française de Pharmacologie et de Thérapeutique ([SFPT](https://sfpt-fr.org/))")
st.title("FAQ médicaments, vaccins et COVID")
st.info("""Cette IA permet une recherche sémantique en langage naturel dans les foires aux questions [médicaments et COVID](https://sfpt-fr.org/covid19) et [vaccins et COVID](https://sfpt-fr.org/faq-vaccins) de la SFPT """)


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Posez moi une question sur médicament et COVID"}
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
    url2s=[
        "https://sfpt-fr.org/component/ifaq/article/01-comment-fonctionne-un-vaccin?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/02-comment-d%C3%A9veloppe-t-on-les-vaccins-en-g%C3%A9n%C3%A9ral?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/03-comment-expliquer-qu%E2%80%99on-mette-si-peu-de-temps-pour-d%C3%A9velopper-les-vaccins-de-la-covid-19-sont-ils-des-vaccins-trop-vite-con%C3%A7us?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/04-combien-de-vaccins-de-la-covid-19-sont-ils-en-d%C3%A9veloppement-aujourd%E2%80%99hui?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/05-quel-est-le-panorama-des-vaccins-de-la-covid-19-pourquoi-y-a-t-il-autant-de-vaccins-en-cours-de-d%C3%A9veloppement?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/06-qu-est-ce-qu-un-vaccin-%C3%A0-arn?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/07-par-quelle-voie-vont-%C3%AAtre-administr%C3%A9s-les-vaccins-de-la-covid?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/08-le-vaccin-de-la-covid-va-t-il-me-prot%C3%A9ger-%C3%A0-100?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/09-les-sujets-%C3%A2g%C3%A9s-seront-ils-prot%C3%A9g%C3%A9s-de-la-m%C3%AAme-fa%C3%A7on-par-le-vaccin?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/10-les-vaccins-pourraient-ils-aggraver-la-covid-19?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/11-a-quels-effets-ind%C3%A9sirables-dois-je-m%E2%80%99attendre-sont-ils-les-m%C3%AAmes-pour-tous-les-vaccins?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/12-comment-va-t-on-v%C3%A9rifier-que-les-vaccins-de-la-covid-19-sont-s%C3%BBrs?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/13-en-pratique,-comment-va-se-d%C3%A9rouler-le-suivi-des-effets-ind%C3%A9sirables-des-vaccins-covid?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/15-comment-se-d%C3%A9roule-la-vaccination-contre-la-covid-19-en-pratique?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/16-pourquoi-faut-il-vacciner-une-large-proportion-de-la-population?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/17-je-suis-sous-un-traitement-pour-une-pathologie-chronique,-puis-je-me-faire-vacciner-contre-la-covid-19?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/18-dans-quelles-cellules-retrouve-t-on-les-arnm-contenus-dans-le-vaccin-apr%C3%A8s-l%E2%80%99injection?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/19-quelles-sont-les-contre-indications-des-vaccins-arnm-contre-la-covid-19?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/20-les-mutations-du-sars-cov-2-remettent-elles-en-question-l%E2%80%99efficacit%C3%A9-des-vaccins%20?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/21-je-suis-enceinte-ou-j%E2%80%99allaite-mon-enfant,-puis-je-me-faire-vacciner?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/22-peut-on-administrer-les-vaccins-contre-la-grippe-et-contre-la-covid-19-le-m%C3%AAme-jour?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/23-quelles-sont-les-donn%C3%A9es-d%E2%80%99efficacit%C3%A9-et-de-s%C3%A9curit%C3%A9-de-la-vaccination-chez-les-enfants-et-les-adolescents?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/24-les-vaccins-ont-ils-un-effet-sur-la-transmission-du-virus?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/25-qu%E2%80%99en-est-il-des-donn%C3%A9es-au-long-cours-de-la-vaccination-en-france?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/26-qu%E2%80%99en-est-il-des-vaccins-arnm-bivalents%20-quel-peut-%C3%AAtre-leur-int%C3%A9r%C3%AAt-dans-la-covid-19?catid=110&Itemid=101",
        "https://sfpt-fr.org/component/ifaq/article/27-quel-est-l%E2%80%99impact-de-la-vaccination-sur-les-%C2%AB-covid-longs-%C2%BB?catid=110&Itemid=101",
     ]

    with st.spinner(text="Lecture des pages pharmacovid de la SFPT ! Un peu de patience SVP."):

        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo", temperature=0.5,
                system_prompt="You are an expert on pharmacology and covid and your job is to answer technical questions.  \
                              Assume that all questions are related to the covid. Keep your answers technical and based on facts – do not hallucinate features. \
                              Give answer in French only."
                )
        )
        SimpleWebPageReader = download_loader("SimpleWebPageReader", custom_path="./tempo")
        loader = SimpleWebPageReader()
        documents = loader.load_data(urls=urls+url2s)
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        return index


index = load_data()

#if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
#    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
if "query_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.query_engine = index.as_query_engine()

if prompt := st.chat_input("Votre question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("En reflexion ..."):
            #response = st.session_state.chat_engine.chat(prompt)
            response = st.session_state.query_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history

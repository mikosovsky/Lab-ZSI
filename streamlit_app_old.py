import streamlit as st
from openai import OpenAI
import random
import time
from chat_openrouter import ChatOpenRouter

selected_model = "mistralai/mistral-7b-instruct:free"
model = ChatOpenRouter(model_name=selected_model)

def answer_question(question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

client = OpenAI(
    api_key=st.secrets["API_KEY"],
    base_url=st.secrets["BASE_URL"]
    )

file_explorer = st.sidebar.file_uploader("Add your RAG files :)", type = "pdf", accept_multiple_files=True)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = client.chat.completions.create(
            model=st.secrets["MODEL"],
            messages=st.session_state.messages
        )
        assistant_response = assistant_response.choices[0].message.content
        # # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
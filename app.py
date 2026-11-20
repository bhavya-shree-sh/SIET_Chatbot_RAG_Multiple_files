import streamlit as st

from rag_engine import load_base, ask_rag


# Page configuration
st.set_page_config(
    page_title="SIET Chatbot",
    page_icon="🎓",
    layout="centered"
)


# College heading
st.title(
    "SHRIDEVI INSTITUTE OF ENGINEERING AND TECHNOLOGY"
)

st.caption(
    "Ask anything about College Information, Courses, FAQs, Rules and Regulations, and Placements."
)

st.caption("RAG Pipeline with Multiple PDF Files")

st.caption("Created by BHAVYASHREE S H")


# Get Groq API key
try:
    mykey = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("Groq API key not found in Streamlit Secrets.")
    st.stop()


# Initialize RAG
@st.cache_resource
def initiate_rag():
    return load_base("college_docs")


try:
    index, embed_model, chunks = initiate_rag()

    st.success("College Knowledge Base Loaded Successfully!")

except Exception as e:
    st.error(f"Error loading documents: {e}")
    st.stop()


# Initialize chat history
if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []


# Display previous messages
for qa in st.session_state.chat_hist:

    with st.chat_message(qa["role"]):
        st.write(qa["content"])


# User input
question = st.chat_input(
    "Ask anything about SIET College..."
)


# Generate answer
if question:

    st.session_state.chat_hist.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            try:
                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key=mykey
                )

                st.write(answer)

                st.session_state.chat_hist.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:
                st.error(f"Error generating answer: {e}")


# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_hist = []
    st.rerun()

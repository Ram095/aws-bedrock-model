import streamlit as st
from dotenv import load_dotenv

from bedrock.bedrock import generate_answer, get_bedrock_client
from bedrock.config import load_settings


def init_bedrock():
    settings = load_settings()
    bedrock_client = get_bedrock_client(settings.aws_region)
    return settings, bedrock_client


def main():
    st.set_page_config(page_title="Chat Playground", page_icon="💬")
    st.title("Chat Playground with AWS Bedrock")
    
    with st.spinner("Loading Bedrock client..."):
        settings, bedrock_client = init_bedrock()
    
    st.caption(f"Chat with {settings.bedrock_model_id} powered by AWS Bedrock")

    st.sidebar.header("Configuration")
    st.sidebar.text(f"Model: {settings.bedrock_model_id}")
    st.sidebar.text(f"Region: {settings.aws_region}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Type your message..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_answer(
                    bedrock_client,
                    settings.bedrock_model_id,
                    user_input,
                    [],  # No context
                    max_tokens=512,
                    temperature=0.7,
                )
                response = (response or "").strip() or "No response text was returned by the model."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    load_dotenv()
    main()

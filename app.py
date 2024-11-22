import os
import openai
import dotenv
import streamlit as st

# Load environment variables
dotenv.load_dotenv()

# Initialize Azure OpenAI client
def init_openai_client():
    return openai.AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
    )

def get_ai_response(client, user_message, history):
    """Get response from Azure OpenAI with Cognitive Search integration"""
    try:
        # Prepare the messages including history
        messages = history + [{"role": "user", "content": user_message}]
        
        completion = client.chat.completions.create(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_ID"),
            messages=messages,
            extra_body={
                "data_sources": [
                    {
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": os.environ["AZURE_AI_SEARCH_ENDPOINT"],
                            "index_name": os.environ["AZURE_AI_SEARCH_INDEX"],
                            "authentication": {
                                "type": "api_key",
                                "key": os.environ["AZURE_AI_SEARCH_API_KEY"],
                            }
                        }
                    }
                ],
            }
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("Healthcare Assistant")
    st.write("Ask me about your health plans and benefits!")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize OpenAI client
    client = init_openai_client()

    # Chat input
    user_message = st.chat_input("Type your message here...")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    if user_message:
        # Display user message
        with st.chat_message("user"):
            st.write(user_message)
        st.session_state.messages.append({"role": "user", "content": user_message})

        # Get and display assistant response
        with st.chat_message("assistant"):
            response = get_ai_response(client, user_message, st.session_state.messages[:-1])
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
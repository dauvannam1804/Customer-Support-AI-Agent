from typing import Annotated, TypedDict, List, Any, cast
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from mem0 import Memory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load biến môi trường
load_dotenv()

# Đọc từ file .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Kiểm tra khóa
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Add your GOOGLE_API_KEY in .env")


# Khởi tạo LLM Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
)

config = {
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.5-flash-lite",
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 1.0
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            # "collection_name": "test",
            "embedding_model_dims": 768, # hardcoded to match the embedding dim of the `embedder` model
        }
    }
}

# Khởi tạo Mem0
mem0 = Memory.from_config(config)

class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    mem0_user_id: str

graph = StateGraph(State)

def chatbot(state: State):
    messages = state["messages"]
    user_id = state["mem0_user_id"]

    try:
        # Retrieve relevant memories
        # mem0.search expects a string query. Message content can be str or list/dict,
        # so normalize it to a string first to satisfy static type checkers.
        last_content = messages[-1].content
        if isinstance(last_content, list):
            # Convert list items (which may be strings or dicts) into a single string
            import json
            query = " ".join(item if isinstance(item, str) else json.dumps(item) for item in last_content)
        else:
            query = str(last_content)

        memories = mem0.search(query, user_id=user_id)

        # Handle dict response format
        memory_list = memories.get('results', [])

        context = "Relevant information from previous conversations:\n"
        for memory in memory_list:
            context += f"- {memory['memory']}\n"

        system_message = SystemMessage(content=f"""You are a helpful customer support assistant. Use the provided context to personalize your responses and remember user preferences and past interactions.
{context}""")

        full_messages = [system_message] + messages
        response = llm.invoke(full_messages)

        # Store the interaction in Mem0
        try:
            interaction = [
                {
                    "role": "user",
                    "content": messages[-1].content
                },
                {
                    "role": "assistant", 
                    "content": response.content
                }
            ]
            result = mem0.add(interaction, user_id=user_id)
            print(f"Memory saved: {len(result.get('results', []))} memories added")
        except Exception as e:
            print(f"Error saving memory: {e}")
            
        return {"messages": [response]}
        
    except Exception as e:
        print(f"Error in chatbot: {e}")
        # Fallback response without memory context
        response = llm.invoke(messages)
        return {"messages": [response]}
    

graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", "chatbot")

compiled_graph = graph.compile()

def run_conversation(user_input: str, mem0_user_id: str):
    config = {"configurable": {"thread_id": mem0_user_id}}
    state = {"messages": [HumanMessage(content=user_input)], "mem0_user_id": mem0_user_id}

    # compiled_graph.stream expects a State TypedDict and a RunnableConfig; cast
    # to satisfy the static type checker while preserving runtime behavior.
    for event in compiled_graph.stream(cast(State, state), cast(Any, config)):
        for value in event.values():
            if value.get("messages"):
                print("Customer Support:", value["messages"][-1].content)
                return
            

if __name__ == "__main__":
    print("Welcome to Customer Support! How can I assist you today?")
    mem0_user_id = "alice"  # You can generate or retrieve this based on your user management system
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Customer Support: Thank you for contacting us. Have a great day!")
            break
        run_conversation(user_input, mem0_user_id)
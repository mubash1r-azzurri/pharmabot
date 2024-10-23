import os
import json
from openai import OpenAI
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
import markdown
from langchain_community.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Query

load_dotenv()

BASE_CHAT_MODEL = 'gpt-3.5-turbo'
BASE_MAX_TOKENS = 200
BASE_TEMPERATURE = 0
BASE_PROMPT = """You are helpful assitant. Behave like a genius, You are too talkative.
You are Given a set of documents you have to answer user on the basis of the these documents.
Never Hallucinate. 
Always try to read context in more depth to understand its meaning and answer to user accordingly."""
def load_or_create_bot_settings():
    bot_settings = {}
    bot_settings["max_chat_history_length"] = 10
    file_path = "database/bot_settings.json"
    try:
        with open(file_path, 'r') as file:
            bot_settings = json.load(file)
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            json.dump(bot_settings, file, indent=4)
    return bot_settings
load_or_create_bot_settings()
def save_bot_settings(data):
    file_path = 'database/bot_settings.json'
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    return data

def list_folders(directory_path):
    try:
        folder_names = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
        if folder_names == []:
            return False
        return folder_names
    except FileNotFoundError:
        # print(f"The directory '{directory_path}' does not exist.")
        return False

app = FastAPI()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)
class ChatHistory(BaseModel):
    data:List

def search_medicine(id: str) -> dict:
    pass


ai_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_medicine",
            "description": "Search for medicine in the medicine database, for example when a customer asks 'I want ASPIRIN' or If Pharmasist want to see if the medicine is in the stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "ID of the Medicine",
                    },
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        }
    }
]
def generate_response(
        previous_history=[],
        vector_db_collection=None,
        chatbot_id="chatpdf",
        language="english",
        bot_settings = {}
    ):
    
    prompt = f"""{bot_settings[chatbot_id]["prompt"]}\n
    {vector_db_collection}\n
    """
    new_messages = [
            {
                "role": "system",
                "content": prompt
            }] + previous_history
    response = client.chat.completions.create(
        model=bot_settings[chatbot_id]["chat_model"],
        messages=new_messages,
        temperature=bot_settings[chatbot_id]["temperature"],
        max_tokens=bot_settings[chatbot_id]["max_tokens"],
    )
    answer = response.choices[0].message.content
    return markdown.markdown(answer)[3:-4]

def file_search(description:str):
    pass

@app.post('/chatbot/{chatbot_id}/chat')
async def chat(
    chatbot_id:str,
    chat_history:ChatHistory,
    language:str=Query('english'),
    query:str=Query(...) 
):
    bot_settings = load_or_create_bot_settings()
    # chat_history = chat_history.data[-bot_settings[chatbot_id]["max_chat_history_length"]:]
    chat_history = chat_history.data
    
    chat_history += [{"role":"user","content":query}] 

    chatbots = list_folders('database')
    if chatbots:
        if chatbot_id not in chatbots:
            error_message = {"status":404,"message":f"No data found for the provided chatbot id."}
            return JSONResponse(content=error_message, status_code=404)  
    embeddings = OpenAIEmbeddings()
    docs = None
    persist_directory = f'database'
    med_indexes = list_folders(persist_directory)
    if med_indexes != False:
        pdfs_vector_db = FAISS.load_local(f'database/{med_indexes[0]}',embeddings,allow_dangerous_deserialization=True)
        
        for index in range(1,len(med_indexes)):
            persist_directory = f'database/{med_indexes[index]}'
            newVectordb= FAISS.load_local(persist_directory, embeddings,allow_dangerous_deserialization=True)
            pdfs_vector_db.merge_from(newVectordb)
    
      
        retriever =pdfs_vector_db.as_retriever(search_type="mmr")
        docs = retriever.get_relevant_documents(query,k=3)
    else:
        error_message = {"status":403,"message":f"No Trained PDFs Found."}
        return JSONResponse(content=error_message, status_code=403)  

    found_pdfs = ""
    for i, doc in enumerate(docs):
        found_pdfs += str(f"{i + 1}. {doc.page_content} \n")

    try:
      
        message = generate_response(
            vector_db_collection=found_pdfs,
            previous_history=chat_history,
            chatbot_id=chatbot_id,
            language=language,
            bot_settings=bot_settings
        )
        chat_history.append({"role":"assistant","content":f"BACKLOG:\n{found_pdfs}\n"+message})

        successful_response = {'status': 200,'message': message, 'memory_backlog': chat_history}
        return JSONResponse(content=successful_response, status_code=200)
    except Exception as e:
        print(str(e))
        error_message = {"status":403,"message":"Something went wrong while Chatting please see system logs."}
        return JSONResponse(content=error_message, status_code=403)


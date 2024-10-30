import os
import json
import requests
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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

BASE_CHAT_MODEL = 'gpt-4-turbo'
BASE_MAX_TOKENS = 1000
BASE_TEMPERATURE = 0
prompt = f"""You are Pharmabot. You are now talking to '[USERNAME]', User Login STATUS is '[LOGIN_STATUS]'.

**PharmaBot** is a friendly, empathetic, and knowledgeable virtual pharmacist, designed to engage users in a human-like conversation. Its primary goal is to provide thoughtful and appropriate medicine recommendations based on the user's symptoms while ensuring a smooth, personalized experience.  You are a chatbot for 'Stilo Pharmacy'. you can only provide details of the medicines in our stock. If user ask for order tracking, call the tool `track_order_status` and make decisions accordingly. if their login status is `FALSE`, prompt them `You need to Login or Sign Up in order to ask this type of questions. [AUTH]`.

Here's how PharmaBot operates:

1. **Gathering Key Information**:  
   Begin by asking a few thoughtful questions to understand the user's symptoms, their severity, and any relevant medical history (e.g., allergies, ongoing medications). **Always ask for the user's age**, as it plays a crucial role in diagnosing and recommending appropriate treatments.

2. **Humanized Conversational Flow**:  
   Maintain a warm, empathetic tone throughout the conversation. Ensure the user feels understood, and adapt to their language and communication style, making the experience feel natural and reassuring.

3. **Medicine Search Process**:  
   After gathering sufficient information, PharmaBot uses the `search_medicine_database_in_natural_language` function to identify suitable medications by passing the gathered description of the condition or medicine name. **Only recommend medicines found in the database** and explain why they are suitable for the user's condition, ensuring clarity about the benefits, dosage, and any other important details. if user ask for more options, search for new medicines with more accurate natural language.

4. **Handling Purchase Requests**:  
   If a user asks to buy a specific medicine, PharmaBot cannot directly search for its availability using the `search_medicine_realtime` tool unless it has the medicine’s ID. First, PharmaBot must gather the name of the medicine and retrieve its details (including the ID) through the `search_medicine_database_in_natural_language` tool. Once the ID is obtained, **PharmaBot can then check for real-time availability and price** using the `search_medicine_realtime` tool and guide the user on how to proceed with the purchase.

5. **Direct Queries**:  
   When a user directly asks for a specific medicine, PharmaBot should skip any diagnostic questions. Instead, it should immediately search for the medicine using the name in the `search_medicine_database_in_natural_language` tool to gather the medicine's details. If the user wishes to purchase, PharmaBot can use the medicine's ID from this search to check its stock and price.

6. **Handling Critical Situations**:  
   If the user's condition seems critical or requires immediate medical attention, gently advise them to seek a professional consultation. Provide a link for booking a consultation if necessary, and ensure that the advice is always responsible and prioritizes the user's well-being.

"""
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
class ChatBody(BaseModel):
    token:str
    data:List

ai_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_medicine_database_in_natural_language",
            "description": "Search for medicines in the medicine database in natural language for RAW information e.g: 'A person has fever etc'",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short description of the disease",
                    },
                },
                "required": ["description"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_medicine_realtime",
            "description": "Search for medicine in the medicine database in realtime, for example when a customer asks 'I want ASPIRIN' or If Pharmasist want to see if the medicine is in the stock",
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
    },
    {
        "type": "function",
        "function": {
            "name": "track_order_status",
            "description": "Track the status of an order using the provided order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "ID of the order to track",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        }
    }
]

def generate_response(
        previous_history=[],
        user = {},
        chatbot_id="chatpdf",
    ):
    bot_settings = load_or_create_bot_settings()
    print(user)
    global prompt
    prompt = prompt.replace("[USERNAME]",user["username"])
    prompt = prompt.replace("[LOGIN_STATUS]",user["login_status"])
    new_messages = [
            {
                "role": "system",
                "content": prompt
            }] + previous_history
    response = client.chat.completions.create(
        model=BASE_CHAT_MODEL,
        messages=new_messages,
        temperature=BASE_TEMPERATURE,
        max_tokens=BASE_MAX_TOKENS,
        tools=ai_tools,
        tool_choice="auto"
    )
    return response

def search_medicine_database_in_natural_language(description:str) -> dict:
    print(f"SEARCHING MEDICINE FOR DESCRIPTION: '{description}'")
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
        docs = retriever.get_relevant_documents(description,k=5)
    else:
        return {"search_results":f"No Results found!"}
          

    found_meds = ""
    for i, doc in enumerate(docs):
        found_meds += str(f"{i + 1}. {doc.page_content} \n")
        
    return {"search_results":found_meds}

def search_medicine_realtime(id: str) -> dict:
    print(f"SEARCHING MEDICINE FOR ID: '{id}'")
    url = f"https://multivendor.doclive.info/api/v1/items/details/{id}"


    headers = {
        "moduleId": "1" 
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        data = {
            "id": data["id"],
            "name": data["name"],
            "stock": data["stock"],
            "price": data["price"],
            "buy_url": data["web_link"]
        }
        return{
            "search_results": str(data)
        }
    else:
        return{
            "search_results": f"Product with ID '{id}' currently not available."
        } 

def track_order_status(order_id: str,token: str) -> dict:
    print(f"TRACKING ORDER STATUS FOR ORDER ID: '{order_id}'")
    return {
            "message": "user need to login"
        }
    # url = f"https://multivendor.doclive.info/api/v1/orders/status/{order_id}"

    # headers = {
    #     "moduleId": "1" 
    # }

    # response = requests.get(url, headers=headers)

    # if response.status_code == 200:
    #     data = response.json()
    #     data = {
    #         "order_id": data["order_id"],
    #         "status": data["status"],
    #         "estimated_delivery": data.get("estimated_delivery", "Not available")
    #     }
    #     return {
    #         "order_status": str(data)
    #     }
    # else:
    #     return {
    #         "order_status": f"Order with ID '{order_id}' currently not available."
    #     }

MAX_CALLS = 1

def function_caller(tool_calls,token):
    messages = []
    for tool in tool_calls:
        if (tool.function.name == "search_medicine_database_in_natural_language"):
            print("I should search for medicines information in the database...")
            results = search_medicine_database_in_natural_language(description=json.loads(tool.function.arguments)["description"])
            messages.append(
                {
                    "role": "assistant",
                    "content": f"""Pharmabot Memory Backlog:
                    Searched For: '{json.loads(tool.function.arguments)["description"]}'
                    Results: {str(results['search_results'])}"""
                }
            )
        elif (tool.function.name == "search_medicine_realtime"):
            print("I should search realtime information of the medicine...")
            results = search_medicine_realtime(id=json.loads(tool.function.arguments)["id"])
            messages.append(
                {
                    "role": "assistant",
                    "content": f"""Pharmabot Memory Backlog:
                    Searched For: '{json.loads(tool.function.arguments)["id"]}'
                    Results: {str(results['search_results'])}"""
                }
            )
        elif (tool.function.name == "track_order_status"):
            print("I should track the order status...")
            results = track_order_status(order_id=json.loads(tool.function.arguments)["order_id"],token=token)
            messages.append(
                {
                    "role": "assistant",
                    "content": f"""Pharmabot Memory Backlog:
                    Searched For: '{json.loads(tool.function.arguments)["order_id"]}'
                    Results: {str(results['message'])}"""
                }
            )
    return messages

def process_user_instruction(chatbot_id,user,chat_history,token):
    num_calls = 0
    messages = chat_history
    # while num_calls < MAX_CALLS:
    response = generate_response(previous_history=chat_history,user=user,chatbot_id=chatbot_id)
    message = response.choices[0].message
    
    print(message)
    try:
        # print(f"\n>> Function call #: {num_calls + 1}\n")
        # print(message.tool_calls)
        messages = messages + function_caller(tool_calls=message.tool_calls,token=token)
        # For the sake of this example, we'll simply add a message to simulate success.
        # Normally, you'd want to call the function here, and append the results to messages.
        # messages.append(
        #     {
        #         "role": "tool",
        #         "content": "success",
        #         "tool_call_id": message.tool_calls[0].id,
        #     }
        # )
        # print(new_messages)
        response2 = client.chat.completions.create(
            model=BASE_CHAT_MODEL,
            temperature=BASE_TEMPERATURE,
            max_tokens=BASE_MAX_TOKENS,
            messages= [{"role":"system","content":prompt}]+messages,
        )
        # print("\n>> Message:\n")
        # print(response2.choices[0])
        # num_calls += 1
        messages.append({"role":"assistant","content": response2.choices[0].message.content})
    except Exception as e:
        print(str(e))
        print("\nJust A normal chit chat with bot..\n")
        # print(message.content)
        messages.append({"role":"assistant","content":message.content})
    
    return messages


@app.post('/chatbot/{chatbot_id}/chat')
async def chat(
    chatbot_id:str,
    chat_body:ChatBody,
    language:str=Query('english'),
    query:str=Query(...) 
):
    print(chat_body.token)
    login_status = "TRUE"
    if chat_body.token == "":
        login_status = "FALSE"

    # bot_settings = load_or_create_bot_settings()
    # chat_history = chat_history.data[-bot_settings[chatbot_id]["max_chat_history_length"]:]
    chat_history = chat_body.data
    print(query)
    chat_history += [{"role":"user","content":query}] 

    user = {
        "username": "Nikki",
        "login_status": login_status
    }
    # try:
    new_messages = process_user_instruction(chatbot_id=chatbot_id,user=user,chat_history=chat_history,token=chat_body.token)
    
    final_response = markdown.markdown(new_messages[-1]["content"])[3:-4]
    if "[AUTH]" not in final_response:
        successful_response = {'status': 200,'message': final_response, 'memory_backlog': new_messages}
        return JSONResponse(content=successful_response, status_code=200)
    else:
        final_response = final_response.replace("[AUTH]","")
        successful_response = {'status': 403,'message': final_response, 'memory_backlog': new_messages}
        return JSONResponse(content=successful_response, status_code=403)
    # except Exception as e:
    #     print(str(e))
    #     error_message = {"status":403,"message":"Something went wrong while Chatting please see system logs."}
    #     return JSONResponse(content=error_message, status_code=403)

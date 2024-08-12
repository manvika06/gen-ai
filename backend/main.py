
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import pandas as pd
from pydantic import BaseModel
import shutil
from pathlib import Path
import asyncio
import pandas as pd
from datetime import datetime
import os
import whisper
from gtts import gTTS
import uuid
import shutil
import threading
import queue
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain_groq import ChatGroq
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
unique_folder = f"./data/temp/db_{uuid.uuid4().hex}"
FRONTEND_URL=os.getenv("FRONTEND_URL")
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  # Specific allowed origin
    allow_credentials=True,
    allow_methods=["*"],  # Allowing all methods
    allow_headers=["*"],  # Allowing all headers
)
alerts_db = []

async def periodic_update_rag_model():
    while True:
        # Load or update the RAG model here
        # For instance, you could reload data from a file or databas
        await update_rag_model()  # Assuming this function does not need to be awaited
        await asyncio.sleep(10)  # Wait for 10 seconds before the next run

@app.on_event("startup")
async def startup_event():
    shutil.rmtree('data/temp', ignore_errors=True)
    asyncio.create_task(periodic_update_rag_model())  # Start the periodic task as soon as the app starts


CSV_DIR = "data"

class SimulationStart(BaseModel):
    file: str

class UserQuery(BaseModel):
    text: str



def fetch_answer_from_rag(query, row):
    # Assuming you have a ready-to-use RAG model
    try:
        print(f"Fetching answer from RAG for query: {query}")
        print(row)
        row_context = '\n'.join([f"{col}: {row[col]}" for col in row.index])
        result = rag({"query": query, "context": row_context})
        print("result ",result)
        answer = result["result"]
        print("answer ",answer)
        return answer
    except Exception as e:
        print(f"Failed to fetch answer from RAG: {e}")
        return "Failed to fetch an answer"

def process_user_query(user_text):
    # This function should be defined to handle the query with the Groq model.
    # Assuming `fetch_answer_from_rag` uses the RAG model to process the query
    print(f"User query: {user_text}")
    # Example: Fetching the answer based on some contextual data
    # You would replace this with the actual data fetching logic
    dummy_row = pd.Series({'ENG_TMP': 75, 'FUEL_LV': 50})  # Example row data
    answer = fetch_answer_from_rag(user_text, dummy_row)
    print(f"Answer: {answer}")
    return answer


@app.post("/api/process-text")
async def process_text(query: UserQuery):
    try:
        
        response = process_user_query(query.text)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}, 500  # Return a JSON error message


@app.get("/api/csv-files")
async def get_csv_files():
    try:
        files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def load_documents(manual_path, alerts_path, data_path):
    global unique_folder
    print("Loading documents...")
    try:
        # Read the manual text
        manual_text = Path(manual_path).read_text()

        combined_text = manual_text

        # Read the alerts if the file exists
        if Path(alerts_path).exists():
            print("Alerts file found.")
            alerts_data = pd.read_csv(alerts_path)
            print(f"Alerts data columns: {alerts_data.columns}")  # Debug print
            
            # Use a more flexible approach to format alerts
            alerts_text = []
            for _, row in alerts_data.iterrows():
                alert_parts = []
                for col in row.index:
                    if pd.notna(row[col]):
                        alert_parts.append(f"{col}: {row[col]}")
                alerts_text.append(" | ".join(alert_parts))
            
            combined_text += f"\n\nAlerts:\n" + "\n".join(alerts_text)

        # Read the processed data if the file exists
        if Path(data_path).exists():
            print("Processed data file found.")
            data_df = pd.read_csv(data_path)
            data_text = data_df.to_string(index=False)
            combined_text += f"\n\nProcessed Data:\n{data_text}"

        document_path = Path("data/temp/combined.md")
        if not document_path.parent.exists():
            document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_text(combined_text)

        # Load and process documents
        loader = UnstructuredMarkdownLoader(document_path)
        loaded_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
        docs = text_splitter.split_documents(loaded_documents)

        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
        unique_folder = f"./data/temp/db_{uuid.uuid4().hex}"
        qdrant = Qdrant.from_documents(docs, embeddings, path=unique_folder, collection_name="document_embeddings")

        retriever = qdrant.as_retriever(search_kwargs={"k": 5})
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

        llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

        prompt_template = """
        You are an AI voice assistant specialized in automobile diagnostics. Based on the provided context,
        analyze the data to identify any potential issues. Give the answer in less than 20 words and keep it crisp.

        Context: {context}
        Question: {question}

        Answer:
        """

        chain_type_kwargs = {
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        }

        rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

        print("Documents loaded successfully.")
        return rag

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # This will print the full stack trace
        if unique_folder and os.path.exists(unique_folder):
            shutil.rmtree(unique_folder)
        return None
    
async def update_rag_model():
    global rag
    try:
        rag = load_documents('data/manual.txt', 'data/temp/alerts.csv', 'data/temp/processed_data.csv')
        print("RAG model updated successfully.")
    except Exception as e:
        print(f"Error updating RAG model: {e}")

@app.post("/api/start-simulation")
async def start_simulation(simulation: SimulationStart):
    global alerts_db
    shutil.rmtree('data/temp', ignore_errors=True)
    os.makedirs('data/temp', exist_ok=True)
    alerts_db = []  # Reset the alerts database
    file_path = os.path.join(CSV_DIR, simulation.file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    
    return {"message": f"Simulation started with file: {simulation.file}"}

def check_thresholds_and_generate_alerts(row, thresholds):
    breaches = []
    for key, value in row.items():
        try:
            value_float = float(str(value).replace('%', '').replace('C', '').replace('RPM', '').replace('kPa', '').replace('g/s', '').replace('°', '').replace('km/h', '').replace('hours', '').strip())
            if key in thresholds:
                if 'min' in thresholds[key] and value_float < thresholds[key]['min']:
                    breaches.append({
                        'Parameter': key,
                        'Threshold': 'below minimum',
                        'Current Value': value_float
                    })
                elif 'max' in thresholds[key] and value_float > thresholds[key]['max']:
                    breaches.append({
                        'Parameter': key,
                        'Threshold': 'above maximum',
                        'Current Value': value_float
                    })
        except ValueError:
            continue
    return breaches


def check_thresholds_and_generate_alerts_for_combo(row,thresholds):
    alerts = {}
    for key, value in row.items():
        try:
            value_float = float(str(value).replace('%', '').replace('C', '').replace('RPM', '').replace('kPa', '').replace('g/s', '').replace('°', '').replace('km/h', '').replace('hours', '').strip())
            if key in thresholds:
                if value_float < thresholds[key]['min'] or value_float > thresholds[key]['max']:
                    alerts[key] = True
                else:
                    alerts[key] = False
            else:
                alerts[key] = False
        except ValueError:
            alerts[key] = False
    print(f"Alerts for row: {alerts}")  # Add this line for debugging
    return alerts


@app.websocket("/ws/{file_name}")
async def websocket_endpoint(websocket: WebSocket, file_name: str):
    global alerts_db
    await websocket.accept()
    try:
        file_path = os.path.join(CSV_DIR, file_name)
        if not os.path.exists(file_path):
            await websocket.send_json({"type": "error", "message": f"File not found: {file_name}"})
            return

        df = pd.read_csv(file_path)
        thresholds = {
            'ENGINE_TEMPERATURE': {'min': 20, 'max': 90},
            'FUEL_LEFT': {'min': 30, 'max': 200},
            # Add other thresholds as needed
        }

        alerts_db = []
        processed_data = []

        for index, row in df.iterrows():
            row_data = row.to_dict()
            
            alerts = check_thresholds_and_generate_alerts_for_combo(row, thresholds)
            combined_data = {
                'data': row_data,
                'alerts': alerts,
            }
            
            await websocket.send_json({"type": "row_data", "data": combined_data})
            
            breaches = check_thresholds_and_generate_alerts(row, thresholds)
            for breach in breaches:
                alert_message = f"{breach['Parameter']} alert: {breach['Current Value']}"
                await websocket.send_json({"type": "alert", "message": alert_message})
                alerts_db.append({
                    'Parameter': breach['Parameter'],
                    'Value': breach['Current Value'],
                    'Threshold': f"{breach['Threshold']} limit",
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            processed_data.append(row_data)

            # Save alerts and processed data to CSV after processing each row
            if alerts_db or processed_data:
                save_alerts_and_data_to_csv(
                    alerts_db, 
                    processed_data, 
                    'data/temp/alerts.csv', 
                    'data/temp/processed_data.csv'
                )
                alerts_db.clear()
                processed_data.clear()

            await asyncio.sleep(4)  # Simulate delay for demonstration

    except Exception as e:
        logging.error(f"Error in WebSocket: {str(e)}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()
def save_alerts_and_data_to_csv(alerts_db, processed_data, alerts_file_path, data_file_path):
    try:
        # Save alerts
        df_alerts = pd.DataFrame(alerts_db)
        print(f"Alerts data columns before saving: {df_alerts.columns}")  # Debug print
        df_alerts.to_csv(alerts_file_path, index=False, mode='a', header=not os.path.exists(alerts_file_path))
        print(f"Alerts saved to {alerts_file_path}.")

        # Save processed data
        df_data = pd.DataFrame(processed_data)
        print(f"Processed data columns before saving: {df_data.columns}")  # Debug print
        df_data.to_csv(data_file_path, index=False, mode='a', header=not os.path.exists(data_file_path))
        print(f"Processed data saved to {data_file_path}.")
    except Exception as e:
        print(f"Error saving alerts and data to CSV: {e}")
        print(f"Alerts data: {alerts_db}")
        print(f"Processed data: {processed_data}")
        import traceback
        traceback.print_exc()  # This will print the full stack trace
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

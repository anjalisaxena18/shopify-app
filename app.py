from flask import Flask, render_template, request, jsonify

import os
import re
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

# Access the API key
api_key = os.getenv('API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


loader = CSVLoader(file_path="data\\products.csv", csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['','ProductId','Title','Description','Vendor','ProductType','Price','ImageURL','ProductURL']
}, encoding= 'utf-8',)

product_document = loader.load()

db = FAISS.from_documents(product_document, embedding = embed_model)
retriever = db.as_retriever()

#######
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is. "
    "Ensure the reformulated question explicitly asks for details including URLs and Image URLs where relevant."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

prompt = (
    '''You are a conversational AI specializing in product recommendations.
    Whenever asked about recommendations include the product URL and image URL, along with any other necessary details, and enhance the details accordingly with your knowledge but within the context.
    Use only the provided context, no external knowledge allowed.
    be a conversational mode do proper conversation
    I want to display this so while displaying imagewith size fix 200 x200 to convert it and answer the question in markdown format.
    always display image markdown as img src tag  
    and display product url as product url and on clicking that product should open
    <context>
    {context}
    </context>
'''
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

document_chain = create_stuff_documents_chain(llm, qa_prompt)
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

####
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


product_urls = pd.read_csv('data/products.csv')
product_url_dict = dict(zip(product_urls['Title'], product_urls['ProductURL']))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    if request.is_json:
        data = request.get_json()
        message = data.get('message')
        if not message:
            return jsonify({"error": "Message is required"}), 400

        md_response_2 = conversational_rag_chain.invoke(
        {"input": message},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
)####
        print("Output:\n", md_response_2)
        response_2 = {"reply": md_response_2['answer']}

        return jsonify(response_2)
    else:
        return jsonify({"error": "Invalid input"}), 400
    

def parse_response(response):
    # Split the response into lines
    lines = response.split('\n')

    # Initialize the list to hold product information
    products = []
    current_product = []

    for line in lines:
        if line.startswith("ProductId"):
            if current_product:
                products.append("\n".join(current_product))
            current_product = [line]
        else:
            current_product.append(line)
    if current_product:
        products.append("\n".join(current_product))

    # Initialize the HTML structure
    html_products = ""

    for product in products:
        try:
            if product.strip():  # Ensure the product string is not empty
                # Extract product information using regular expressions
                product_id_match = re.search(r'ProductId: (\d+)', product)
                title_match = re.search(r'Title: (.+)', product)
                description_match = re.search(r'Description: (.+)', product)
                vendor_match = re.search(r'Vendor: (.+)', product)
                product_type_match = re.search(r'ProductType: (.+)', product)
                price_match = re.search(r'Price: ([\d.]+)', product)
                image_url_match = re.search(r'ImageURL: (https?://[^\s]+)', product)
                product_url_match = re.search(r'ProductURL: (https?://[^\s]+)(?=:)', product)

                if all([product_id_match, title_match, description_match, vendor_match, product_type_match, price_match, image_url_match, product_url_match]):
                    product_id = product_id_match.group(1).strip()
                    title = title_match.group(1).strip()
                    description = description_match.group(1).strip()
                    vendor = vendor_match.group(1).strip()
                    product_type = product_type_match.group(1).strip()
                    price = price_match.group(1).strip()
                    image_url = image_url_match.group(1).strip()
                    product_url = product_url_match.group(1).strip()

                    # Generate HTML code for the product
                    product_html = f"""
                    <div class="card border-0 mb-4">
                        <img src="{image_url}" class="card-img-top d-block w-100px" />
                        <div class="card-body">
                            <h5 class="card-title">{title}</h5>
                            <p class="card-text">{description}</p>
                            <p class="card-text">Vendor: {vendor}</p>
                            <p class="card-text">Type: {product_type}</p>
                            <p class="card-text">Price: Rs. {price}</p>
                            <a href="{product_url}" class="btn btn-dark px-5 rounded-4 mt-3">Buy Now</a>
                        </div>
                    </div>
                    """
                    
                    # Append HTML code for the product to the overall HTML code
                    html_products += product_html
                else:
                    print(f"Product information not found for: {product}")

        except Exception as e:
            print(f"Error parsing product: {e}")

    # Return the final HTML response
    html_response = html_products
    return html_response

@app.route('/send_recommendation', methods=['POST'])
def send_recommendation():
    if request.is_json:
        data = request.get_json()
        message = data.get('message')
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        md_response = db.similarity_search(message)
        html_response = ''
        for response in md_response:
            r = response.page_content
            html_response += r
        html_response = parse_response(html_response)
        print('HTML Response:\n',html_response)
        response = {"reply": html_response}

        return jsonify(response)
    else:
        return jsonify({"error": "Invalid input"}), 400

@app.route('/get_product_url', methods=['POST'])
def get_product_url():
    data = request.get_json()
    product_name = data.get('product_name')
    url = product_url_dict.get(product_name)
    if url:
        return jsonify({"url": url})
    else:
        return jsonify({"error": "Product URL not found"}), 404

@app.route('/reset_chat_engine', methods=['POST'])
def reset_chat_engine():
    chat_engine_reset()
    return jsonify({"reply": "History Dumped Successfully"})

def chat_engine_reset():

    return "History Dumped Successfully"


if __name__ == '__main__':
    app.run(debug = False)

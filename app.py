# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from langchain.vectorstores import Chroma
# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, logging
# from langchain.llms import HuggingFacePipeline
# from langchain.chains import RetrievalQA
# from langchain.embeddings import SentenceTransformerEmbeddings
# import torch
#
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes
#
# # Load the pre-trained Chroma DB
# persist_directory = "flask_backend\chroma_persist_new"  # Replace with the path to your Chroma DB folder
# embeddings_model_name = "multi-qa-mpnet-base-dot-v1"
# embeddings = SentenceTransformerEmbeddings(model_name=embeddings_model_name)
# db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#
# # Load the HuggingFace model
# checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# logging.set_verbosity_info()
# base_model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint,
#     device_map="auto",
#     torch_dtype=torch.float32,
#     offload_folder="./offload_weights"  # Specify a folder
# )
#
#
# pipe = pipeline(
#     "text2text-generation",
#     model=base_model,
#     tokenizer=tokenizer,
#     max_length=512,
#     do_sample=False,
#     temperature=0.3,
#     top_p=0.95,
# )
# local_llm = HuggingFacePipeline(pipeline=pipe)
#
# # Create the QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=local_llm,
#     chain_type="stuff",
#     retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
#     return_source_documents=True,
# )
#
# @app.route("/query", methods=["POST"])
# def handle_query():
#     try:
#         data = request.json
#         query = data.get("query", "")
#         if not query:
#             return jsonify({"error": "Query not provided"}), 400
#
#         # Run the query through the QA chain
#         response = qa_chain({"query": query})
#         result = response.get("result", "No result found.")
#         return jsonify({"response": result})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.vectorstores import Chroma
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, logging
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained Chroma DB
persist_directory = "chroma_persist_new"  # Adjust as needed
embeddings_model_name = "multi-qa-mpnet-base-dot-v1"
embeddings = SentenceTransformerEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Load the HuggingFace model
checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
logging.set_verbosity_info()
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype=torch.float32,
    offload_folder="./offload_weights"
)

pipe = pipeline(
    "text2text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=False,  # Deterministic output
    temperature=0.3,
    top_p=0.95,
)

local_llm = HuggingFacePipeline(pipeline=pipe)

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    return_source_documents=True,
)

@app.route("/query", methods=["GET", "POST"])
def handle_query():
    global last_query, last_response
    if request.method == "POST":
        data = request.json
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Empty query provided"}), 400

        response = qa_chain({"query": query})
        last_query = query
        last_response = response.get("result", "No result found.")
        return jsonify({"response": last_response})

    # For GET requests, return the last response
    elif request.method == "GET":
        return jsonify({"response": last_response or "No response yet."})


# @app.route("/query", methods=["POST"])
# def handle_query():
#     try:
#         data = request.json
#         query = data.get("query", "").strip()
#         if not query:
#             return jsonify({"error": "Empty query provided"}), 400
#
#         response = qa_chain({"query": query})
#         result = response.get("result", "No result found.")
#         source_docs = response.get("source_documents", [])
#
#         sources = [
#             {"content": doc.page_content, "metadata": doc.metadata} for doc in source_docs
#         ]
#
#         return jsonify({"response": result, "sources": sources})
#     except Exception as e:
#         app.logger.error(f"Error processing query: {str(e)}")
#         return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


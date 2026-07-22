# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
"""import torch
import transformers
import sys
def print_hi(name):


    print(torch.__version__)
    print(transformers.__version__)

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    )

    inputs = tokenizer(
        "hello world",
        return_tensors="pt"
    )

    print("before forward")

    with torch.no_grad():
        output = model(**inputs)

    print("after forward")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    print(sys.executable)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/"""

import streamlit as st
from retriever import retrieve
from reranker import rerank


st.title("AI Financial Research Assistant")

question = st.chat_input("Ask a financial question")

if question:
    print(question)
    retrieved_chunks = retrieve(question)
    print(retrieved_chunks)
    answer = rerank(question, retrieved_chunks)
    print(answer)
    st.chat_message("assistant").write(answer)

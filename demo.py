import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG
from dotenv import load_dotenv
load_dotenv()
def main():
    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.", #0
        "George Rankin is a politician.",#1
        "Thomas Marwick is a politician.",#2
        "Cinderella attended the royal ball.",#3
        "The prince used the lost glass slipper to search the kingdom.",#4
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",#5
        "Erik Hort's birthplace is Montebello.",#6
        "Marina is bom in Minsk.",#7
        "Montebello is a part of Rockland County."#8
    ]

    save_dir = 'outputs'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    embedding_model_name = 'text-embedding-3-large'  # Embedding model name (NV-Embed, GritLM or Contriever for now)

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name)

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    # For Evaluation
    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
         "The prince used the lost glass slipper to search the kingdom.",
         "When the slipper fit perfectly, Cinderella was reunited with the prince."],
        ["Erik Hort's birthplace is Montebello.",
         "Montebello is a part of Rockland County."]
    ]

    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers))

    print("#################################################################")
    # retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=2)
    # print(retrieval_results)
    print("#################################################################")
    # qa_results = hipporag.rag_qa(retrieval_results)
    # print(qa_results)

    # # Combined Retrieval & QA
    # rag_results = hipporag.rag_qa(queries=queries)

if __name__ == "__main__":
    main()

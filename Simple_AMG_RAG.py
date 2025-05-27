import json
import os
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.utils.json import OutputParserException
from langchain.docstore.document import Document
from langchain_community.tools import DuckDuckGoSearchResults
from decouple import config
from typing import List
from duckduckgo_search.exceptions import RatelimitException
import wikipedia
import networkx as nx
from transformers import pipeline
from langgraph.graph import END, StateGraph, START
import time
from typing_extensions import TypedDict
from neo4j import GraphDatabase
from create_VDB import MedicalQAChromaDB
import wikipediaapi
from langchain_ollama import ChatOllama
import requests
import time
from xml.etree import ElementTree as ET

class PubMedAPI:
    def __init__(self, api_key=config('pubmed_api')):
        self.api_key = api_key
        self.requests_made = 0
        self.last_request_time = time.time()

    def throttle(self):
        """Ensures API call rate limits are respected."""
        max_requests = 10 if self.api_key else 3
        time_since_last_request = time.time() - self.last_request_time
        if self.requests_made >= max_requests and time_since_last_request < 1:
            time.sleep(1 - time_since_last_request)
            self.requests_made = 0
        self.last_request_time = time.time()
        self.requests_made += 1

    def search_pubmed(self, query, max_results=5):
        """Search PubMed for the given query and return PubMed IDs."""
        self.throttle()
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": max_results,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        return [id_elem.text for id_elem in root.findall(".//Id")]

    def fetch_articles(self, pmids):
        """Fetch raw articles for the given PubMed IDs."""
        if not pmids:
            return []

        self.throttle()
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "text",
            "rettype": "abstract",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.text.split("\n\n")  # Return articles as raw text, split by double newline

class QAChainProcessor:
    def __init__(self):
        # self.llm = ChatOllama(
        #     model="llama3.2",
        #     temperature=0.0,
        #     num_predict=200,
        #     format="json"
        # )
        self.llm= ChatOpenAI(model="gpt-4o", temperature=0, api_key=config('OPENAI_API_KEY'))
        self.pubmed_api = PubMedAPI(api_key=config("pubmed_api"))
        self.max_entity_size = 2
        self.max_doc_search = 3
        
 
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = Chroma(collection_name="medical_qa", embedding_function=self.embedding_model)
        self.db = MedicalQAChromaDB()

        self.qa_chain = self.get_qa_chain()
        self.cot_chain = self.get_cot_chain()
        self.search_list = self.search_list()
        self.workflow = StateGraph(self.GraphState)
        self.create_graph()

    class GraphState(TypedDict):
        question: str
        thoughts: List[str]
        generation: str
        searchs: List[str]
        documents: List[str]
        search_list: List[str]

    def format_docs(self, docs):
        return "\n\n".join(doc[0].page_content for doc in docs)

    def get_qa_chain(self):
        response_schemas = [
            ResponseSchema(
                name="finalresponse", 
                description="final response can be one of 'A', 'B', 'C', 'D', 'E' options or if none, return 'NAN'", 
                type="string"
            )
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template="""You are answering the question below based only on the provided documents and thoughts.
            Use this evidence to select the best answer.

            Facts:
            {documents} 
            
            Reasoning:
            {thoughts} 
            
            Question: {question}
            Respond in valid JSON format with the answer using these instructions: 
            {format_instructions}""",
            input_variables=["question", "documents", "thoughts"],
            partial_variables={"format_instructions": format_instructions},
        )
        return prompt | self.llm | parser

    def generate(self, state):
        question = state["question"]
        documents = state["documents"]
        thoughts = state["thoughts"]
        generation = self.qa_chain.invoke({"documents": documents, "thoughts": thoughts, "question": question})
        self.last_answer = generation['finalresponse']
        return {"documents": documents, "generation": generation['finalresponse']}

    def get_cot_chain(self):
        response_schemas = [
            ResponseSchema(
                name="thoughts",
                description="The generated reasoning and chain of thought to answer the question.",
                type="string",
            ),
        ]

        cot_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = cot_parser.get_format_instructions()

        prompt = """You are creating a step by step thoughts to logically answer the question
        based on the following context:

        Context:
        {context}
        
        And the Search Context:
        {search_context}
        Question: {question}
        Provide a detailed reasoning chain and final thought in JSON format using these instructions: 
        {format_instructions}"""

        cot_prompt = PromptTemplate(
            template=prompt,
            input_variables=["question", "context", "search_context"],
            partial_variables={"format_instructions": format_instructions},
        )
        return cot_prompt | self.llm | cot_parser

    def generate_cot(self, state):
        question = state["question"]
        documents = state["documents"]
        search_contexts = state.get('searchs', [])
        self.search_results = search_contexts

        individual_thoughts = []
        
        for search_item in search_contexts:
            try:
                cot = self.cot_chain.invoke({
                    "question": question,
                    "context": documents,
                    "search_context": search_item["content"]
                })
                individual_thoughts.append({"search_item": search_item, "thought": cot["thoughts"]})
            except Exception as e:
                print(f"Error generating CoT for search item '{search_item}': {e}")
                individual_thoughts.append({"search_item": search_item, "thought": "Error in CoT generation"})

        self.cot = individual_thoughts
        return {"thoughts": individual_thoughts, "question": question}

    def search_list(self):
        response_schemas = [
            ResponseSchema(
                name="search_phrases",
                description="A list of search-friendly medical phrases  no more than three, extracted from the question, optimized for Pubmed search",
                type="array"
            )
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        prompt = """
        Question: {question}

        Create a short list no more than three search-friendly medical phrases optimized for Pubmed search in valid JSON format.
        Use these instructions: {format_instructions}
        """

        _prompt = PromptTemplate(
            template=prompt,
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions}
        )
        return _prompt | self.llm | parser

    def gen_search_list(self, state):
        question = state.get("question", "")
        
        try:
            search_list = self.search_list.invoke({"question": question})
            return {
                "search_list": search_list.get("search_phrases", [])
                if search_list and "search_phrases" in search_list
                else []
            }
        except Exception as e:
            print(f"Error generating search list: {e}")
            return {"search_list": [" "]}

    def retrieve_vdb(self, state):
        question = state["question"]
        documents = self.db.main(mode="query", query_text=question)
        for item in self.options_list:
            documents.extend(self.db.main(mode="query", query_text=item,n_results=1))
        formatted_documents = self.format_docs(documents)
        return {"documents": formatted_documents[::-1]}
    def get_pubmed_results(self, state):
        print("---SEARCH NODE (PubMed)---")
        items = state.get("search_list", [])
        items.extend(self.options_list)
        self.search_items = items  # Store search items
        print(self.search_items)
        individual_results = []

        if not items:
            print("No items in search list.")
            return {"searchs": "No search items found"}

        # Case-insensitive keywords for filtering articles
        filter_keywords = [
            "author", "doi", "pmid", "conflict of interest", "copyright",
            "comment on", "editorial", "department of", "university", "college", "institute"
        ]

        # Search and retrieve results for each item
        for item in items:
            try:
                pmids = self.pubmed_api.search_pubmed(item, max_results=self.max_entity_size)
                articles = self.pubmed_api.fetch_articles(pmids)
                
                # Extract only the main content (abstracts or main text) from articles
                filtered_articles = []
                for article in articles:
                    main_content = []
                    for line in article.split("\n"):
                        if line.strip() and not any(keyword.lower() in line.lower() for keyword in filter_keywords):
                            main_content.append(line.strip())
                    if main_content:
                        filtered_articles.append(" ".join(main_content))

                combined_content = "\n\n".join(filtered_articles)  # Combine all articles for the item
                individual_results.append({"search_item": item, "content": combined_content})
            except Exception as e:
                print(f"Error while searching PubMed for '{item}': {e}")
                individual_results.append({"search_item": item, "content": "Error in retrieving content"})

        return {"searchs": individual_results}


    def get_wiki_results(self, state):
        print("---SEARCH NODE (Wikipedia)---")
        items = state.get("search_list", [])
        items.extend(self.options_list)
        self.search_items = items  # Storing the items to be used later if necessary
        print(self.search_items)
        individual_results = []

        if not items:
            print("No items in search list.")
            return {"searchs": "No search items found"}

        # Perform the search and retrieve results for each item
        for item in items:
            content = ""
            
            try:
                results = wikipedia.search(item, results=self.max_doc_search)  # Search for each item
                for result_title in results:
                    try:
                        summary = wikipedia.summary(result_title, sentences=self.max_doc_search)
                        content += summary + "\n"
                    except wikipedia.exceptions.DisambiguationError as e:
                        print(f"Disambiguation error for {result_title}: {e.options}")
                    except wikipedia.exceptions.PageError:
                        print(f"Page not found for title {result_title}")
                
                # Append each search item and its content as a dictionary
                individual_results.append({"search_item": item, "content": content})
            
            except Exception as e:
                print(f"Error while searching Wikipedia for '{item}': {e}")
                individual_results.append({"search_item": item, "content": "Error in retrieving content"})

        # Return the individual search results as a list of dictionaries
        return {"searchs": individual_results}

    def process_question(self, question_data):
        question = question_data["question"]
        options = question_data.get("options", {})
        options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
        self.options_list = [f"{value}" for key, value in options.items()]
        self.question=question
        # Run the pipeline
        _ = self.run_pipeline(question + options_str)
        
        # Retrieve documents from the vector database
        retrieved_docs = self.retrieve_vdb({"question": question})["documents"]

        return {
            "question": question,
            "options": options_str,
            "expected_answer": question_data["answer"],
            "model_answer": self.last_answer if hasattr(self, "last_answer") else "N/A",
            "answer_idx": question_data["answer_idx"],
            "cot_list": self.cot if hasattr(self, "cot") else [],
            "search_results": self.search_results if hasattr(self, "search_results") else "N/A",
            "documents": retrieved_docs,
            "meta_info": question_data.get("meta_info", ""),
            "search_items": self.search_items if hasattr(self, "search_items") else "N/A"
        }


    @staticmethod
    def load_questions(jsonl_file):
        with open(jsonl_file, "r",encoding='utf-8') as file:
            return [json.loads(line.strip()) for line in file]

    @staticmethod
    def load_results_from_csv(output_csv):
        return pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame(columns=["question", "expected_answer", "model_answer", "meta_info", "answer_idx", "q_idx"])

    @staticmethod
    def save_results_to_csv(results, output_csv_base, batch_size=10):
        for i in range(0, len(results), batch_size):
            # Convert `cot_list`, `search_results`, `search_items`, and `documents` to JSON strings for each result
            for result in results[i:i + batch_size]:
                result['cot_list'] = json.dumps(result['cot_list'])
                result['search_results'] = json.dumps(result['search_results']) if isinstance(result['search_results'], list) else result['search_results']
                result['search_items'] = json.dumps(result['search_items']) if isinstance(result['search_items'], list) else result['search_items']
                result['documents'] = json.dumps(result['documents'])

            # Define the output file name for each batch
            output_csv = f"{output_csv_base}"
            
            # Append or create a new CSV file
            if os.path.exists(output_csv):
                df_existing = pd.read_csv(output_csv)
                df_combined = pd.concat([df_existing, pd.DataFrame(results[i:i + batch_size])], ignore_index=True)
                df_combined.to_csv(output_csv, index=False)
            else:
                pd.DataFrame(results[i:i + batch_size]).to_csv(output_csv, index=False)


    def create_graph(self):
        self.workflow.add_node("retrieve", self.retrieve_vdb)
        self.workflow.add_node("gen_search_list", self.gen_search_list)
        self.workflow.add_node("searches", self.get_pubmed_results)
        self.workflow.add_node("generate_cot", self.generate_cot)
        self.workflow.add_node("final_answer", self.generate)

        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "gen_search_list")
        self.workflow.add_edge("gen_search_list", "searches")
        self.workflow.add_edge("searches", "generate_cot")
        self.workflow.add_edge("generate_cot", "final_answer")
        self.workflow.add_edge("final_answer", END)
        self.app = self.workflow.compile()

    def run_pipeline(self, question):
        inputs = {"question": question}
        for output in self.app.stream(inputs, {"recursion_limit": 50}):
            pass
        return output

    def main(self, jsonl_file, output_csv):
        questions = self.load_questions(jsonl_file)
        processed_df = self.load_results_from_csv(output_csv)
        processed_question_ids = processed_df['q_idx'].unique().astype(int) if not processed_df.empty else []
        print('---')
        print(processed_question_ids)
        results = []
        for idx, question_data in enumerate(questions):
            print(f"processing Q#{idx}/{len(questions)}")
           
            if idx in processed_question_ids:
                # print(f"Skipping processed question: {question_id}")
                pass
            else:
                result = self.process_question(question_data)
                result['q_idx']=idx
                results.append(result)

                if len(results) % 1 == 0:
                    self.save_results_to_csv(results, output_csv)
                    results = []

        if results:
            self.save_results_to_csv(results, output_csv)
        print(f"Results saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    jsonl_file = "dataset/MEDQA/questions/US/test.jsonl"
    output_csv = "results/AMG_pubmed_test.csv"
    
    processor = QAChainProcessor()
    processor.main(jsonl_file, output_csv)


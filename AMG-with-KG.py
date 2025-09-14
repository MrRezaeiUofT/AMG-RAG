"""
AMG-RAG: Autonomous Medical Knowledge Graph RAG System
Complete implementation with dynamic KG generation and medical QA
"""

import json
import os
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import networkx as nx
from langchain_openai import ChatOpenAI
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import requests
from xml.etree import ElementTree as ET
import wikipedia
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from decouple import config
# Configuration - Replace with your API keys
OPENAI_API_KEY =config('OPENAI_API_KEY')  # Replace with your key
PUBMED_API_KEY = config('pubmed_api')  # Optional, can be None

@dataclass
class MedicalEntity:
    """Represents a medical entity in the knowledge graph"""
    name: str
    description: str
    entity_type: str  # drug, disease, symptom, treatment, etc.
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)

@dataclass
class MedicalRelation:
    """Represents a relationship between medical entities"""
    source: str
    target: str
    relation_type: str
    confidence: float
    evidence: str
    sources: List[str] = field(default_factory=list)

class MedicalKnowledgeGraph:
    """Dynamic Medical Knowledge Graph with confidence scoring"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities = {}
        self.relations = []
        
    def add_entity(self, entity: MedicalEntity):
        """Add a medical entity to the graph"""
        self.entities[entity.name] = entity
        self.graph.add_node(
            entity.name,
            description=entity.description,
            entity_type=entity.entity_type,
            confidence=entity.confidence,
            sources=entity.sources
        )
        
    def add_relation(self, relation: MedicalRelation):
        """Add a relationship between entities"""
        self.relations.append(relation)
        self.graph.add_edge(
            relation.source,
            relation.target,
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            evidence=relation.evidence,
            sources=relation.sources
        )
        
    def get_connected_nodes(self, node_name: str, confidence_threshold: float = 0.5):
        """Get nodes connected to a given node with confidence above threshold"""
        connected = []
        if node_name in self.graph:
            for neighbor in self.graph.neighbors(node_name):
                edge_data = self.graph[node_name][neighbor]
                if edge_data.get('confidence', 0) >= confidence_threshold:
                    connected.append({
                        'node': neighbor,
                        'relation': edge_data.get('relation_type'),
                        'confidence': edge_data.get('confidence'),
                        'evidence': edge_data.get('evidence')
                    })
        return connected
    
    def explore_path(self, start_node: str, max_depth: int = 3, 
                    confidence_threshold: float = 0.5):
        """Explore paths from a starting node with confidence propagation"""
        paths = []
        visited = set()
        
        def dfs(node, path, accumulated_confidence, depth):
            if depth > max_depth or node in visited:
                return
            
            visited.add(node)
            
            if len(path) > 0:
                paths.append({
                    'path': path.copy(),
                    'confidence': accumulated_confidence,
                    'final_node': node
                })
            
            for neighbor_data in self.get_connected_nodes(node, confidence_threshold):
                neighbor = neighbor_data['node']
                new_confidence = accumulated_confidence * neighbor_data['confidence']
                
                if new_confidence >= confidence_threshold:
                    new_path = path + [(node, neighbor, neighbor_data['relation'])]
                    dfs(neighbor, new_path, new_confidence, depth + 1)
            
            visited.remove(node)
        
        dfs(start_node, [], 1.0, 0)
        return paths

class PubMedSearcher:
    """PubMed API wrapper for medical literature search"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
    def search(self, query: str, max_results: int = 3) -> List[str]:
        """Search PubMed and return article abstracts"""
        # Search for PMIDs
        search_url = f"{self.base_url}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": max_results
        }
        if self.api_key:
            search_params["api_key"] = self.api_key
            
        try:
            response = requests.get(search_url, params=search_params, timeout=10)
            root = ET.fromstring(response.text)
            pmids = [id_elem.text for id_elem in root.findall(".//Id")]
            
            if not pmids:
                return []
            
            # Fetch abstracts
            fetch_url = f"{self.base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "text",
                "rettype": "abstract"
            }
            if self.api_key:
                fetch_params["api_key"] = self.api_key
                
            response = requests.get(fetch_url, params=fetch_params, timeout=10)
            articles = response.text.split("\n\n")
            
            # Clean and return abstracts
            abstracts = []
            for article in articles:
                lines = article.split("\n")
                abstract_lines = [line for line in lines if line.strip() 
                                and not any(skip in line.lower() for skip in 
                                          ["author", "doi", "pmid", "copyright"])]
                if abstract_lines:
                    abstracts.append(" ".join(abstract_lines))
                    
            return abstracts
            
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []

class AMG_RAG_System:
    """Main AMG-RAG system for medical question answering"""
    
    def __init__(self, use_openai: bool = True, openai_key: str = None):
        # Initialize LLM
        if use_openai and openai_key:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Use gpt-4o-mini for cost efficiency
                temperature=0.0,
                api_key=openai_key
            )
        else:
            # Fallback to local Ollama if available
            if ChatOllama is not None:
                self.llm = ChatOllama(
                    model="llama3.2",
                    temperature=0.0
                )
            else:
                raise ImportError("Neither OpenAI API key provided nor Ollama available. Please install langchain_ollama or provide OpenAI API key.")
            
        # Initialize components
        self.kg = MedicalKnowledgeGraph()
        self.pubmed = PubMedSearcher(api_key=PUBMED_API_KEY)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = Chroma(
            collection_name="medical_qa",
            embedding_function=self.embeddings
        )
        
        # Initialize chains
        self._setup_chains()
        
    def _setup_chains(self):
        """Setup LLM chains for various tasks"""
        
        # Enhanced medical entity extraction with relevance scoring
        entity_schemas = [
            ResponseSchema(
                name="entities",
                description="List of medical entities (diseases, drugs, symptoms, treatments)",
                type="array"
            ),
            ResponseSchema(
                name="scores",
                description="Relevance scores (1-10) for each entity based on importance to the question",
                type="array"
            ),
            ResponseSchema(
                name="descriptions",
                description="Brief descriptions of each entity in the context of the question",
                type="array"
            )
        ]
        entity_parser = StructuredOutputParser.from_response_schemas(entity_schemas)
        
        self.entity_extractor = PromptTemplate(
            template="""Extract all medical entities from this question and options with relevance scoring.
            Include diseases, drugs, symptoms, treatments, and medical concepts.
            
            Question: {question}
            Options: {options}
            Context: {context}
            
            For each entity, provide:
            1. Entity name
            2. Relevance score (1-10): 10=directly related to question, 7-9=moderately relevant, 4-6=weakly relevant, 1-3=minimally relevant
            3. Brief description in context of the question
            
            Return in JSON format:
            {format_instructions}""",
            input_variables=["question", "options", "context"],
            partial_variables={"format_instructions": entity_parser.get_format_instructions()}
        ) | self.llm | entity_parser
        
        # Enhanced relation extraction with bidirectional analysis
        relation_schemas = [
            ResponseSchema(
                name="relationships",
                description="List of relationship types between entities",
                type="array"
            ),
            ResponseSchema(
                name="scores",
                description="Confidence scores (1-10) for each relationship",
                type="array"
            ),
            ResponseSchema(
                name="evidence",
                description="Evidence supporting each relationship",
                type="array"
            )
        ]
        relation_parser = StructuredOutputParser.from_response_schemas(relation_schemas)
        
        self.relation_extractor = PromptTemplate(
            template="""Analyze the medical relationships between these entities based on the context.
            
            Entity 1: {entity1}
            Description 1: {desc1}
            
            Entity 2: {entity2}  
            Description 2: {desc2}
            
            Context: {context}
            
            Provide relationships in this exact JSON format:
            {{
                "relationships": [
                    {{
                        "entityA": "{entity1}",
                        "entityB": "{entity2}",
                        "relationship_type": "relationship_type_here",
                        "confidence_score": 8,
                        "evidence": "brief evidence here"
                    }},
                    {{
                        "entityA": "{entity2}",
                        "entityB": "{entity1}",
                        "relationship_type": "relationship_type_here",
                        "confidence_score": 7,
                        "evidence": "brief evidence here"
                    }}
                ]
            }}
            
            Use medical relationship types like: treats, causes, symptom_of, risk_factor_for, contraindicated_with, differential_diagnosis, etc.
            Confidence scores: 10=strong evidence, 7-9=moderate evidence, 4-6=weak evidence, 1-3=minimal evidence
            
            Return ONLY the JSON, no other text:""",
            input_variables=["entity1", "desc1", "entity2", "desc2", "context"],
            partial_variables={"format_instructions": relation_parser.get_format_instructions()}
        ) | self.llm | relation_parser
        
        # Entity summarization chain
        summary_schemas = [
            ResponseSchema(
                name="summaries",
                description="Concise summaries for each entity based on context",
                type="array"
            ),
            ResponseSchema(
                name="scores",
                description="Relevance scores (1-10) for each summary",
                type="array"
            )
        ]
        summary_parser = StructuredOutputParser.from_response_schemas(summary_schemas)
        
        self.summary_chain = PromptTemplate(
            template="""Generate concise and relevant summaries for each medical entity based on the given context.
            
            Entities: {entities}
            Context: {context}
            
            For each entity, provide:
            1. A concise summary (2-3 sentences) focusing on relevance to the medical question
            2. Relevance score (1-10): 10=directly relevant, 7-9=moderately relevant, 4-6=weakly relevant, 1-3=minimally relevant
            
            Return in JSON format:
            {format_instructions}""",
            input_variables=["entities", "context"],
            partial_variables={"format_instructions": summary_parser.get_format_instructions()}
        ) | self.llm | summary_parser
        
        # Chain of thought reasoning
        cot_schemas = [
            ResponseSchema(
                name="reasoning",
                description="Step-by-step medical reasoning",
                type="string"
            )
        ]
        cot_parser = StructuredOutputParser.from_response_schemas(cot_schemas)
        
        self.cot_chain = PromptTemplate(
            template="""Based on the medical knowledge graph information and search results,
            provide step-by-step reasoning for this medical question.
            
            Question: {question}
            
            Graph Knowledge:
            {graph_context}
            
            Search Results:
            {search_context}
            
            Provide detailed medical reasoning:
            {format_instructions}""",
            input_variables=["question", "graph_context", "search_context"],
            partial_variables={"format_instructions": cot_parser.get_format_instructions()}
        ) | self.llm | cot_parser
        
        # Final answer generation
        answer_schemas = [
            ResponseSchema(
                name="answer",
                description="Final answer (A, B, C, D, or E)",
                type="string"
            ),
            ResponseSchema(
                name="confidence",
                description="Confidence in the answer (0-1)",
                type="number"
            ),
            ResponseSchema(
                name="explanation",
                description="Brief explanation",
                type="string"
            )
        ]
        answer_parser = StructuredOutputParser.from_response_schemas(answer_schemas)
        
        self.answer_chain = PromptTemplate(
            template="""Based on the reasoning and evidence, select the best answer.
            
            Question: {question}
            Options: {options}
            
            Reasoning:
            {reasoning}
            
            Evidence:
            {evidence}
            
            Select the best answer (A, B, C, D, or E):
            {format_instructions}""",
            input_variables=["question", "options", "reasoning", "evidence"],
            partial_variables={"format_instructions": answer_parser.get_format_instructions()}
        ) | self.llm | answer_parser
        
    def build_knowledge_graph(self, question: str, options: Dict[str, str]) -> None:
        """Build a dynamic knowledge graph for the question with enhanced entity extraction"""
        
        # Prepare context for entity extraction
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        full_text = question + " " + " ".join(options.values())
        
        # Search for additional context
        search_query = question + " " + " ".join(list(options.values())[:3])
        search_results = self.pubmed.search(search_query, max_results=3)
        context = "\n".join(search_results) if search_results else ""
        
        # Extract medical entities with relevance scoring
        try:
            entities_result = self.entity_extractor.invoke({
                "question": question,
                "options": options_text,
                "context": context
            })
            entities = entities_result.get("entities", [])
            scores = entities_result.get("scores", [])
            descriptions = entities_result.get("descriptions", [])
        except Exception as e:
            print(f"Entity extraction error: {e}")
            entities = list(options.values())[:3]  # Fallback to options
            scores = [5] * len(entities)  # Default moderate relevance
            descriptions = [f"Medical concept: {entity}" for entity in entities]
        
        print(f"Extracted entities: {entities}")
        print(f"Relevance scores: {scores}")
        
        # Add entities to graph with relevance-based confidence
        for i, entity in enumerate(entities[:8]):  # Limit to 8 entities
            # Search PubMed for additional information
            abstracts = self.pubmed.search(entity, max_results=2)
            
            # Search Wikipedia as fallback
            wiki_content = ""
            try:
                wiki_results = wikipedia.search(entity, results=1)
                if wiki_results:
                    wiki_content = wikipedia.summary(wiki_results[0], sentences=3)
            except:
                pass
            
            # Combine sources with LLM-generated description
            llm_description = descriptions[i] if i < len(descriptions) else f"Medical entity: {entity}"
            external_description = " ".join(abstracts) if abstracts else wiki_content
            combined_description = f"{llm_description}. {external_description}" if external_description else llm_description
            
            # Calculate confidence based on relevance score and external sources
            relevance_score = scores[i] if i < len(scores) else 5
            confidence = min(1.0, (relevance_score / 10.0) + (0.2 if abstracts else 0.1))
            
            # Add entity to graph
            med_entity = MedicalEntity(
                name=entity,
                description=combined_description[:500],  # Limit description length
                entity_type="medical_concept",
                confidence=confidence,
                sources=["PubMed", "Wikipedia"] if abstracts else ["Wikipedia"]
            )
            self.kg.add_entity(med_entity)
        
        # Extract relationships between entities with enhanced analysis
        entity_list = list(self.kg.entities.keys())
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i+1:]:
                try:
                    # Get descriptions
                    desc1 = self.kg.entities[entity1].description
                    desc2 = self.kg.entities[entity2].description
                    
                    # Enhanced context for relationship analysis
                    relationship_context = f"{question}\n\nOptions: {options_text}\n\nSearch Results: {context}"
                    
                    # Extract relationships with bidirectional analysis
                    relation_result = self.relation_extractor.invoke({
                        "entity1": entity1,
                        "desc1": desc1,
                        "entity2": entity2,
                        "desc2": desc2,
                        "context": relationship_context
                    })
                    
                    # Process relationships from the structured JSON response
                    relationships = relation_result.get("relationships", [])
                    
                    # Process each relationship in the list
                    for rel in relationships:
                        if isinstance(rel, dict):
                            rel_type = rel.get("relationship_type", "related_to")
                            confidence = rel.get("confidence_score", 5) / 10.0
                            evidence = rel.get("evidence", "")
                            entity_a = rel.get("entityA", "")
                            entity_b = rel.get("entityB", "")
                            
                            # Create the relationship
                            relation = MedicalRelation(
                                source=entity_a,
                                target=entity_b,
                                relation_type=rel_type,
                                confidence=confidence,
                                evidence=evidence,
                                sources=["LLM Analysis"]
                            )
                            self.kg.add_relation(relation)
                    
                except Exception as e:
                    print(f"Relation extraction error for {entity1}-{entity2}: {e}")
        
        # Generate entity summaries for better context
        self._generate_entity_summaries(question, context)
                    
    def _generate_entity_summaries(self, question: str, context: str) -> None:
        """Generate enhanced summaries for entities in the knowledge graph"""
        if not self.kg.entities:
            return
            
        try:
            entities_list = list(self.kg.entities.keys())
            summary_result = self.summary_chain.invoke({
                "entities": entities_list,
                "context": f"Question: {question}\n\nContext: {context}"
            })
            
            summaries = summary_result.get("summaries", [])
            scores = summary_result.get("scores", [])
            
            # Update entity descriptions with enhanced summaries
            for i, entity_name in enumerate(entities_list):
                if i < len(summaries) and i < len(scores):
                    # Combine original description with enhanced summary
                    original_desc = self.kg.entities[entity_name].description
                    enhanced_summary = summaries[i]
                    relevance_score = scores[i]
                    
                    # Update description with enhanced summary
                    updated_description = f"{original_desc}\n\nEnhanced Summary: {enhanced_summary}"
                    
                    # Update confidence based on summary relevance
                    current_confidence = self.kg.entities[entity_name].confidence
                    summary_confidence = min(1.0, relevance_score / 10.0)
                    updated_confidence = min(1.0, (current_confidence + summary_confidence) / 2)
                    
                    # Update the entity
                    self.kg.entities[entity_name].description = updated_description[:500]
                    self.kg.entities[entity_name].confidence = updated_confidence
                    
        except Exception as e:
            print(f"Entity summarization error: {e}")
                    
    def reason_with_graph(self, question: str, options: Dict[str, str]) -> Dict[str, Any]:
        """Perform reasoning using the knowledge graph"""
        
        # Explore graph paths for each entity
        graph_context = []
        for entity in list(self.kg.entities.keys())[:3]:
            # Get connected nodes
            connections = self.kg.get_connected_nodes(entity, confidence_threshold=0.3)
            
            # Explore paths
            paths = self.kg.explore_path(entity, max_depth=2, confidence_threshold=0.3)
            
            context = f"Entity: {entity}\n"
            context += f"Description: {self.kg.entities[entity].description[:200]}\n"
            
            if connections:
                context += "Direct connections:\n"
                for conn in connections[:3]:
                    context += f"  - {conn['relation']} -> {conn['node']} (confidence: {conn['confidence']:.2f})\n"
            
            if paths:
                context += "Reasoning paths:\n"
                for path_data in paths[:2]:
                    path_str = " -> ".join([f"{p[0]} [{p[2]}]" for p in path_data['path']])
                    if path_str:
                        context += f"  - {path_str} -> {path_data['final_node']} (confidence: {path_data['confidence']:.2f})\n"
            
            graph_context.append(context)
        
        # Search for additional evidence
        search_query = question + " " + " ".join(list(self.kg.entities.keys())[:3])
        search_results = self.pubmed.search(search_query, max_results=2)
        search_context = "\n".join(search_results) if search_results else "No additional search results found."
        
        # Generate chain of thought reasoning
        try:
            cot_result = self.cot_chain.invoke({
                "question": question,
                "graph_context": "\n\n".join(graph_context),
                "search_context": search_context
            })
            reasoning = cot_result.get("reasoning", "Unable to generate reasoning")
        except Exception as e:
            print(f"CoT generation error: {e}")
            reasoning = "Error in reasoning generation"
        
        # Generate final answer
        options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
        evidence = "\n".join(graph_context[:2])
        
        try:
            answer_result = self.answer_chain.invoke({
                "question": question,
                "options": options_str,
                "reasoning": reasoning,
                "evidence": evidence
            })
            
            return {
                "answer": answer_result.get("answer", "Unable to determine"),
                "confidence": answer_result.get("confidence", 0.0),
                "explanation": answer_result.get("explanation", ""),
                "reasoning": reasoning,
                "graph_context": graph_context,
                "search_context": search_context
            }
        except Exception as e:
            print(f"Answer generation error: {e}")
            return {
                "answer": "Error",
                "confidence": 0.0,
                "explanation": str(e),
                "reasoning": reasoning,
                "graph_context": graph_context,
                "search_context": search_context
            }
    
    def answer_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main pipeline to answer a medical question"""
        
        question = question_data["question"]
        options = question_data.get("options", {})
        
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print(f"Options: {options}")
        print(f"{'='*50}\n")
        
        # Step 1: Build knowledge graph
        print("Step 1: Building knowledge graph...")
        self.build_knowledge_graph(question, options)
        print(f"Graph has {len(self.kg.entities)} entities and {len(self.kg.relations)} relations")
        
        # Step 2: Reason with graph
        print("\nStep 2: Reasoning with graph...")
        result = self.reason_with_graph(question, options)
        
        # Add metadata
        result["question"] = question
        result["options"] = options
        result["expected_answer"] = question_data.get("answer", "Unknown")
        result["graph_stats"] = {
            "num_entities": len(self.kg.entities),
            "num_relations": len(self.kg.relations)
        }
        
        return result

def load_medqa_sample():
    """Load a sample from MEDQA dataset"""
    # Sample MEDQA question
    sample = {
        "question": "A 45-year-old man presents to the emergency department with severe chest pain that started 2 hours ago. The pain is substernal, crushing in nature, and radiates to his left arm. He has a history of hypertension and diabetes mellitus. His father died of a myocardial infarction at age 50. On examination, he is diaphoretic and in distress. His blood pressure is 150/90 mmHg, pulse is 110/min, and respirations are 22/min. An ECG shows ST-segment elevation in leads II, III, and aVF. Which of the following is the most likely diagnosis?",
        "options": {
            "A": "Unstable angina",
            "B": "Acute inferior wall myocardial infarction",
            "C": "Acute anterior wall myocardial infarction", 
            "D": "Aortic dissection",
            "E": "Pulmonary embolism"
        },
        "answer": "B",
        "answer_idx": 1,
        "meta_info": "This is a cardiology question testing knowledge of myocardial infarction presentation and ECG findings."
    }
    return sample

def main():
    """Main execution function"""
    
    print("AMG-RAG Medical QA System")
    print("="*50)
    
    # Initialize system
    print("Initializing AMG-RAG system...")
    
    # Set to True and provide key to use OpenAI, False for Ollama
    USE_OPENAI = True  # Set to False to use Ollama instead
    
    if USE_OPENAI:
        if OPENAI_API_KEY == "your-openai-api-key":
            print("Warning: Please set your OpenAI API key in the code")
            print("Falling back to Ollama (make sure Ollama is running)")
            USE_OPENAI = False
    
    system = AMG_RAG_System(use_openai=USE_OPENAI, openai_key=OPENAI_API_KEY if USE_OPENAI else None)
    
    # Load sample question
    print("\nLoading MEDQA sample question...")
    question_data = load_medqa_sample()
    
 
    
    result = system.answer_question(question_data)
    
  
    
    # Display results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Question: {result['question'][:100]}...")
    print(f"\nOptions:")
    for k, v in result['options'].items():
        print(f"  {k}: {v}")
    
    print(f"\nExpected Answer: {result['expected_answer']}")
    print(f"Model Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"\nExplanation: {result['explanation']}")
    
    print(f"\nGraph Statistics:")
    print(f"  - Entities: {result['graph_stats']['num_entities']}")
    print(f"  - Relations: {result['graph_stats']['num_relations']}")
    
    print(f"\nReasoning Chain:")
    print(result['reasoning'][:500] + "..." if len(result['reasoning']) > 500 else result['reasoning'])
    

    
    # Visualize graph structure (text-based)
    print("\n" + "="*50)
    print("KNOWLEDGE GRAPH STRUCTURE")
    print("="*50)
    
    for entity_name, entity in list(system.kg.entities.items())[:5]:
        print(f"\n📌 {entity_name}")
        print(f"   Type: {entity.entity_type}")
        print(f"   Description: {entity.description[:100]}...")
        
        connections = system.kg.get_connected_nodes(entity_name)
        if connections:
            print("   Connections:")
            for conn in connections[:3]:
                print(f"     → {conn['relation']} → {conn['node']} (conf: {conn['confidence']:.2f})")

if __name__ == "__main__":
    main()
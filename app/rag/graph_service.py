
import logging
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from llama_index.llms.gemini import Gemini
from . import config

class GraphService:
    def __init__(self):
        self.driver = None
        self.llm = Gemini(model="models/gemini-2.5-flash", api_key=config.GOOGLE_API_KEY, temperature=0.1)
        
        if config.USE_NEO4J:
            try:
                self.driver = GraphDatabase.driver(
                    config.NEO4J_URI,
                    auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
                )
                logging.info("Neo4j connection established")
            except Exception as e:
                logging.warning(f"Neo4j connection failed: {e}. Graph features disabled.")
                self.driver = None
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def extract_topics_and_relations(self, text: str, filename: str) -> Dict:
        """
        Use Gemini to extract topics and their relationships from text.
        Returns: {"topics": [...], "relationships": [(from, to, type), ...]}
        """
        if not self.driver:
            return {"topics": [], "relationships": []}
        
        prompt = f"""Analyze this educational content and extract:
1. Key topics/concepts (as concise names, e.g., "Neural Networks", "Backpropagation")
2. Prerequisite relationships (which topics require knowledge of others)

Content:
{text[:2000]}  

Respond in JSON format:
{{
  "topics": ["Topic1", "Topic2", ...],
  "prerequisites": [{{"topic": "AdvancedTopic", "requires": "BasicTopic"}}, ...]
}}
"""
        
        try:
            response = self.llm.complete(prompt)
            import json
            import re
            
            # Extract JSON from response
            match = re.search(r'\{.*\}', str(response), re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                topics = data.get("topics", [])
                prereqs = data.get("prerequisites", [])
                
                # Store in Neo4j
                self._store_graph_data(topics, prereqs, filename)
                
                return {"topics": topics, "relationships": prereqs}
        except Exception as e:
            logging.error(f"Topic extraction failed: {e}")
        
        return {"topics": [], "relationships": []}
    
    def _store_graph_data(self, topics: List[str], prerequisites: List[Dict], source: str):
        """Store topics and relationships in Neo4j"""
        if not self.driver:
            return
        
        with self.driver.session() as session:
            # Create topics
            for topic in topics:
                session.run(
                    "MERGE (t:Topic {name: $name}) "
                    "ON CREATE SET t.source = $source "
                    "ON MATCH SET t.source = t.source + ', ' + $source",
                    name=topic, source=source
                )
            
            # Create prerequisite relationships
            for prereq in prerequisites:
                session.run(
                    "MATCH (t:Topic {name: $topic}) "
                    "MATCH (p:Topic {name: $requires}) "
                    "MERGE (t)-[:REQUIRES]->(p)",
                    topic=prereq.get("topic"), requires=prereq.get("requires")
                )
    
    def find_prerequisites(self, topic: str) -> List[str]:
        """
        Find all prerequisites for a given topic.
        Returns list of prerequisite topics.
        """
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run(
                "MATCH (t:Topic {name: $topic})-[:REQUIRES*]->(p:Topic) "
                "RETURN DISTINCT p.name as prereq",
                topic=topic
            )
            return [record["prereq"] for record in result]
    
    def check_knowledge_gaps(self, query: str) -> Dict[str, List[str]]:
        """
        Analyze query to identify topics and check for missing prerequisites.
        Returns: {"mentioned_topics": [...], "missing_prerequisites": [...]}
        """
        if not self.driver:
            return {"mentioned_topics": [], "missing_prerequisites": []}
        
        # Extract topics from query using LLM
        prompt = f"Extract key technical topics/concepts from this query (return as JSON array): {query}"
        try:
            response = self.llm.complete(prompt)
            import json
            import re
            match = re.search(r'\[.*\]', str(response), re.DOTALL)
            if match:
                mentioned_topics = json.loads(match.group(0))
            else:
                mentioned_topics = []
        except:
            mentioned_topics = []
        
        # Find prerequisites
        all_prereqs = set()
        for topic in mentioned_topics:
            prereqs = self.find_prerequisites(topic)
            all_prereqs.update(prereqs)
        
        # Check which prerequisites are NOT mentioned in the query
        missing = [p for p in all_prereqs if p not in mentioned_topics]
        
        return {
            "mentioned_topics": mentioned_topics,
            "missing_prerequisites": missing
        }

    def get_graph_data(self, limit: int = 50) -> Dict:
        """
        Retrieve graph nodes and edges for visualization.
        Returns: {"nodes": [{"id": "Name", "group": 1}, ...], "links": [{"source": "A", "target": "B"}, ...]}
        """
        if not self.driver:
            return {"nodes": [], "links": []}
            
        with self.driver.session() as session:
            # Fetch nodes
            result_nodes = session.run(f"MATCH (n:Topic) RETURN n.name as name LIMIT {limit}")
            nodes = [{"id": r["name"], "group": 1} for r in result_nodes]
            
            # Fetch edges
            result_edges = session.run(f"MATCH (a:Topic)-[r:REQUIRES]->(b:Topic) RETURN a.name as source, b.name as target LIMIT {limit}")
            links = [{"source": r["source"], "target": r["target"]} for r in result_edges]
            
            return {"nodes": nodes, "links": links}

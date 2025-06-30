# Import necessary libraries
from neo4j import GraphDatabase
import pandas as pd

# Define connection parameters
uri = "bolt://localhost:7687"  # Update if needed
user = "neo4j"
password = "12345678"

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

# Function to execute Cypher queries
def execute_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]

# Close the driver connection
def close_driver():
    driver.close()
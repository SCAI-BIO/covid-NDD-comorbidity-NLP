import webbrowser

# Your Neo4j AuraDB connection details
NEO4J_URI = "neo4j+s://09f8d4e9.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "a8axV58SA-bajOWDieMaYFcf_U6NhG929g3atbsJQxg"

# Construct the URL for Neo4j Browser with prefilled connection details
neo4j_browser_url = f"https://browser.neo4j.io/?connectURL={NEO4J_URI}&username={NEO4J_USER}&password={NEO4J_PASSWORD}"

# Open Neo4j Browser in the default web browser
webbrowser.open(neo4j_browser_url)

print("Neo4j Browser has been opened with your credentials.")

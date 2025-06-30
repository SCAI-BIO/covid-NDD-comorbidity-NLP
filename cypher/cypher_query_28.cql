# graph density general query

""" 
// Calculate total number of nodes
MATCH (n)
WITH count(n) as nodeCount

// Calculate total number of relationships
MATCH ()-[r]->()
WITH nodeCount, count(r) as edgeCount

// Calculate maximum possible edges for a directed graph
// For undirected graph, divide by 2
WITH nodeCount, edgeCount, nodeCount * (nodeCount - 1) as maxPossibleEdges

// Calculate density
// Density = actual edges / possible edges
WITH 
    nodeCount as nodes,
    edgeCount as edges,
    maxPossibleEdges as possibleEdges,
    toFloat(edgeCount) / (nodeCount * (nodeCount - 1)) as density

RETURN {
    numberOfNodes: nodes,
    numberOfEdges: edges,
    maxPossibleEdges: possibleEdges,
    graphDensity: density
} as graphMetrics
"""
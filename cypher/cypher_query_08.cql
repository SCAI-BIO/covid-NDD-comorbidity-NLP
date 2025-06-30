#node and realtionship tzpe distributions for primekg:
"""
MATCH (n)-[r]->(m)
WHERE any(term IN ["covid", "alzheimer", "neurodegeneration", "parkinson"] 
           WHERE n.name =~ ('(?i).*' + term + '.*') 
              OR m.name =~ ('(?i).*' + term + '.*'))
WITH DISTINCT n AS node, type(r) AS relationship_type
WITH labels(node) AS node_labels, relationship_type, count(node) AS node_count
RETURN node_labels, relationship_type, node_count
ORDER BY node_count DESC
""" 


# degree centrality for primekg

""""
MATCH (n)-[r]->(m)
WHERE any(term IN ["covid", "alzheimer", "neurodegeneration", "parkinson"] 
           WHERE n.name =~ ('(?i).*' + term + '.*') 
              OR m.name =~ ('(?i).*' + term + '.*'))

// Calculate out-degree and in-degree in a single pass
WITH n AS node, count(r) AS out_degree, collect(m) AS targets
UNWIND targets AS target
WITH node, out_degree, target
OPTIONAL MATCH (target)<-[r2]-()
WITH node, out_degree, count(r2) AS in_degree

// Calculate total degree and normalized degree centrality
WITH node, out_degree, in_degree, (out_degree + in_degree) AS total_degree
RETURN 
    node.name AS Node,
    labels(node) AS NodeLabels,
    total_degree,
    out_degree,
    in_degree,
    (total_degree * 1.0 / (1000 - 1)) AS normalized_degree_centrality  // Substitute 1000 with the actual total node count
ORDER BY normalized_degree_centrality DESC

"""
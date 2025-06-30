"""WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords,
     ['disease', 'familial', 'susceptibility', 'to', 'with', 'spastic', 'paraparesis'] AS commonWords // Define common words to be ignored

// Step 1: Match nodes and clean up names by removing punctuation and common words
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH d, 
     apoc.text.regreplace(toLower(d.name), "[^a-zA-Z0-9\\s]", "") AS cleanName, // Remove punctuation and convert to lowercase
     commonWords

// Step 2: Remove common words like 'disease' and trim spaces
WITH d, reduce(name = trim(cleanName), word IN commonWords | apoc.text.replace(name, word, '')) AS finalName

// Step 3: Keep only the base name before the first number (if any) or extra characters
WITH d, apoc.text.regexGroups(finalName, "(alzheimer|parkinson|neurodegenerative|covid|neurodegeneration)") AS baseName

// Step 4: Group nodes by the cleaned and normalized base name
WITH baseName[0] AS GroupName, collect(d) AS similar_nodes

// Step 5: Return distinct groups and their nodes
RETURN 
    GroupName, 
    [node IN similar_nodes | node.name] AS NodeNames
ORDER BY GroupName"""
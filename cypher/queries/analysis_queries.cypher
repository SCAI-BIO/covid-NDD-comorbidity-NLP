// COVID-NDD Comorbidity Analysis - Custom Cypher Queries
// Repository: https://github.com/SCAI-BIO/covid-NDD-comorbidity-NLP
// Description: Collection of Cypher queries for knowledge graph analysis

// ============================================================================
// PART 1: SCHEMA CREATION AND DATA LOADING
// ============================================================================

// Create constraints and indexes for better performance
CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE;
CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.id IS UNIQUE;
CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (pw:Pathway) REQUIRE pw.id IS UNIQUE;

// Create indexes for frequently queried properties
CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name);
CREATE INDEX gene_symbol IF NOT EXISTS FOR (g:Gene) ON (g.symbol);
CREATE INDEX drug_name IF NOT EXISTS FOR (dr:Drug) ON (dr.name);

// ============================================================================
// PART 2: COVID-19 AND NDD ENTITY IDENTIFICATION
// ============================================================================

// Query 1: Identify COVID-19 related diseases
MATCH (d:Disease)
WHERE d.name =~ '(?i).*(covid|sars.cov|coronavirus).*'
   OR d.synonyms =~ '(?i).*(covid|sars.cov|coronavirus).*'
   OR d.id IN ['MONDO:0100096', 'DOID:0080600', 'HP:0033677']
RETURN d.id, d.name, d.synonyms
ORDER BY d.name;

// Query 2: Identify neurodegenerative diseases
MATCH (d:Disease)
WHERE d.name =~ '(?i).*(alzheimer|parkinson|huntington|als|amyotrophic|dementia|neurodegenerat).*'
   OR d.synonyms =~ '(?i).*(alzheimer|parkinson|huntington|als|amyotrophic|dementia|neurodegenerat).*'
   OR d.id IN ['MONDO:0004975', 'MONDO:0005180', 'MONDO:0007739', 'MONDO:0008199']
RETURN d.id, d.name, d.synonyms
ORDER BY d.name;

// ============================================================================
// PART 3: COMORBIDITY PATHWAY DISCOVERY
// ============================================================================

// Query 3: Find direct associations between COVID-19 and NDDs
MATCH (covid:Disease)-[r1]-(intermediate)-[r2]-(ndd:Disease)
WHERE covid.name =~ '(?i).*covid.*'
  AND ndd.name =~ '(?i).*(alzheimer|parkinson|dementia).*'
  AND type(r1) IN ['ASSOCIATED_WITH', 'CAUSES', 'PREDISPOSES_TO']
  AND type(r2) IN ['ASSOCIATED_WITH', 'CAUSES', 'PREDISPOSES_TO']
RETURN covid.name, type(r1), intermediate, type(r2), ndd.name, 
       r1.evidence_count, r2.evidence_count
ORDER BY r1.evidence_count DESC, r2.evidence_count DESC
LIMIT 50;

// Query 4: Find shared genetic factors between COVID-19 and NDDs
MATCH (covid:Disease)-[:ASSOCIATED_WITH]-(g:Gene)-[:ASSOCIATED_WITH]-(ndd:Disease)
WHERE covid.name =~ '(?i).*covid.*'
  AND ndd.name =~ '(?i).*(alzheimer|parkinson|huntington|als|dementia).*'
RETURN covid.name, g.symbol, g.name, ndd.name, 
       COUNT(*) as pathway_count
ORDER BY pathway_count DESC
LIMIT 25;

// Query 5: Identify shared protein targets
MATCH (covid:Disease)-[:TARGETS]-(p:Protein)-[:TARGETS]-(ndd:Disease)
WHERE covid.name =~ '(?i).*covid.*'
  AND ndd.name =~ '(?i).*(neurodegenerat|alzheimer|parkinson).*'
RETURN covid.name, p.name, p.function, ndd.name,
       p.uniprot_id
ORDER BY p.name;

// ============================================================================
// PART 4: DRUG REPURPOSING OPPORTUNITIES
// ============================================================================

// Query 6: Find drugs that target both COVID-19 and NDD pathways
MATCH (drug:Drug)-[:TREATS|:TARGETS]-(covid:Disease),
      (drug)-[:TREATS|:TARGETS]-(ndd:Disease)
WHERE covid.name =~ '(?i).*covid.*'
  AND ndd.name =~ '(?i).*(alzheimer|parkinson|dementia).*'
RETURN drug.name, drug.drugbank_id, 
       collect(DISTINCT covid.name) as covid_indications,
       collect(DISTINCT ndd.name) as ndd_indications
ORDER BY drug.name;

// Query 7: Identify potential drug repurposing candidates
MATCH (ndd_drug:Drug)-[:TREATS]-(ndd:Disease),
      (covid_target:Protein)<-[:TARGETS]-(ndd_drug)
WHERE ndd.name =~ '(?i).*(alzheimer|parkinson|dementia).*'
  AND EXISTS {
    MATCH (covid:Disease)-[:ASSOCIATED_WITH]-(covid_target)
    WHERE covid.name =~ '(?i).*covid.*'
  }
RETURN ndd_drug.name, ndd_drug.drugbank_id, ndd.name, 
       covid_target.name, covid_target.function
ORDER BY ndd_drug.name;

// ============================================================================
// PART 5: PATHWAY ENRICHMENT ANALYSIS
// ============================================================================

// Query 8: Analyze shared biological pathways
MATCH (covid:Disease)-[:ASSOCIATED_WITH]-(g1:Gene)-[:PARTICIPATES_IN]-(pathway:Pathway),
      (ndd:Disease)-[:ASSOCIATED_WITH]-(g2:Gene)-[:PARTICIPATES_IN]-(pathway)
WHERE covid.name =~ '(?i).*covid.*'
  AND ndd.name =~ '(?i).*(alzheimer|parkinson|dementia).*'
RETURN pathway.name, pathway.id, 
       COUNT(DISTINCT g1) as covid_genes,
       COUNT(DISTINCT g2) as ndd_genes,
       COUNT(DISTINCT g1) + COUNT(DISTINCT g2) as total_genes
ORDER BY total_genes DESC
LIMIT 20;

// Query 9: Find inflammation-related connections
MATCH (covid:Disease)-[*1..3]-(inflammation:Pathway)-[*1..3]-(ndd:Disease)
WHERE covid.name =~ '(?i).*covid.*'
  AND ndd.name =~ '(?i).*(alzheimer|parkinson|dementia).*'
  AND inflammation.name =~ '(?i).*(inflammat|cytokine|immune|neuroinflam).*'
RETURN covid.name, inflammation.name, ndd.name,
       length(shortestPath((covid)-[*]-(ndd))) as path_length
ORDER BY path_length
LIMIT 15;

// ============================================================================
// PART 6: EVIDENCE AGGREGATION AND SCORING
// ============================================================================

// Query 10: Calculate comorbidity evidence scores
MATCH (covid:Disease)-[r1*1..2]-(intermediate)-[r2*1..2]-(ndd:Disease)
WHERE covid.name =~ '(?i).*covid.*'
  AND ndd.name =~ '(?i).*(alzheimer|parkinson|dementia).*'
WITH covid, ndd, 
     REDUCE(score = 0, rel IN r1 | score + COALESCE(rel.confidence, 0.5)) +
     REDUCE(score = 0, rel IN r2 | score + COALESCE(rel.confidence, 0.5)) as pathway_score,
     length(r1) + length(r2) as path_length
RETURN covid.name, ndd.name, 
       pathway_score / path_length as normalized_score,
       path_length,
       COUNT(*) as pathway_count
ORDER BY normalized_score DESC, pathway_count DESC
LIMIT 30;

// ============================================================================
// PART 7: LITERATURE MINING SUPPORT QUERIES
// ============================================================================

// Query 11: Find publications supporting comorbidity hypotheses
MATCH (covid:Disease)-[:MENTIONED_IN]-(pub:Publication)-[:MENTIONS]-(ndd:Disease)
WHERE covid.name =~ '(?i).*covid.*'
  AND ndd.name =~ '(?i).*(alzheimer|parkinson|dementia).*'
RETURN pub.pmid, pub.title, pub.journal, pub.year,
       covid.name, ndd.name
ORDER BY pub.year DESC
LIMIT 20;

// Query 12: Extract supporting evidence for specific hypotheses
MATCH (covid:Disease {name: 'COVID-19'})-[r*1..3]-(ndd:Disease)
WHERE ndd.name =~ '(?i).*alzheimer.*'
WITH covid, ndd, r
UNWIND r as relationship
MATCH (relationship)-[:SUPPORTED_BY]-(evidence:Evidence)
RETURN covid.name, ndd.name, 
       type(relationship), evidence.source, evidence.confidence,
       evidence.pmid, evidence.study_type
ORDER BY evidence.confidence DESC;

// ============================================================================
// PART 8: GRAPH STATISTICS AND QUALITY METRICS
// ============================================================================

// Query 13: Calculate graph statistics
MATCH (n)
RETURN labels(n)[0] as node_type, COUNT(n) as count
ORDER BY count DESC;

// Query 14: Assess evidence quality
MATCH ()-[r]-()
WHERE EXISTS(r.evidence_count)
RETURN type(r) as relationship_type, 
       AVG(r.evidence_count) as avg_evidence,
       MIN(r.evidence_count) as min_evidence,
       MAX(r.evidence_count) as max_evidence,
       COUNT(r) as total_relationships
ORDER BY avg_evidence DESC;

// ============================================================================
// PART 9: EXPORT QUERIES FOR ANALYSIS
// ============================================================================

// Query 15: Export comorbidity network for external analysis
MATCH (covid:Disease)-[r*1..2]-(ndd:Disease)
WHERE covid.name =~ '(?i).*covid.*'
  AND ndd.name =~ '(?i).*(alzheimer|parkinson|dementia).*'
WITH covid, ndd, r
RETURN covid.id as covid_id, covid.name as covid_name,
       ndd.id as ndd_id, ndd.name as ndd_name,
       length(r) as path_length,
       [rel in r | type(rel)] as relationship_types,
       [rel in r | COALESCE(rel.confidence, 0.5)] as confidences;

// ============================================================================
// PART 10: HYPOTHESIS GENERATION QUERIES
// ============================================================================

// Query 16: Generate novel comorbidity hypotheses
MATCH (covid:Disease)-[:ASSOCIATED_WITH]-(g1:Gene)-[:INTERACTS_WITH]-(g2:Gene)-[:ASSOCIATED_WITH]-(ndd:Disease)
WHERE covid.name =~ '(?i).*covid.*'
  AND ndd.name =~ '(?i).*(alzheimer|parkinson|dementia).*'
  AND NOT EXISTS {
    MATCH (covid)-[*1..2]-(ndd)
  }
RETURN covid.name, g1.symbol, g2.symbol, ndd.name,
       'Potential novel comorbidity via gene interaction' as hypothesis_type
ORDER BY g1.symbol, g2.symbol
LIMIT 10;
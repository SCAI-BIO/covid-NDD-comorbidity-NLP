// Graph Schema Definition for COVID-NDD Comorbidity Analysis
// Creates the complete database schema with constraints and indexes

// ============================================================================
// NODE CONSTRAINTS (Unique identifiers)
// ============================================================================

CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE;
CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.id IS UNIQUE;
CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (pw:Pathway) REQUIRE pw.id IS UNIQUE;
CREATE CONSTRAINT publication_id IF NOT EXISTS FOR (pub:Publication) REQUIRE pub.pmid IS UNIQUE;
CREATE CONSTRAINT chemical_id IF NOT EXISTS FOR (c:Chemical) REQUIRE c.id IS UNIQUE;

// ============================================================================
// PERFORMANCE INDEXES
// ============================================================================

// Text search indexes
CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name);
CREATE INDEX gene_symbol IF NOT EXISTS FOR (g:Gene) ON (g.symbol);
CREATE INDEX protein_name IF NOT EXISTS FOR (p:Protein) ON (p.name);
CREATE INDEX drug_name IF NOT EXISTS FOR (dr:Drug) ON (dr.name);
CREATE INDEX pathway_name IF NOT EXISTS FOR (pw:Pathway) ON (pw.name);

// Composite indexes for frequent queries
CREATE INDEX disease_type_name IF NOT EXISTS FOR (d:Disease) ON (d.type, d.name);
CREATE INDEX gene_symbol_chromosome IF NOT EXISTS FOR (g:Gene) ON (g.symbol, g.chromosome);

// Full-text search indexes
CREATE FULLTEXT INDEX disease_fulltext IF NOT EXISTS FOR (d:Disease) ON EACH [d.name, d.synonyms, d.description];
CREATE FULLTEXT INDEX publication_fulltext IF NOT EXISTS FOR (p:Publication) ON EACH [p.title, p.abstract];
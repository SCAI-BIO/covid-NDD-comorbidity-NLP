
import pandas as pd
import numpy as np
import random
from statsmodels.stats.multitest import multipletests
import networkx as nx
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

def analyze_complete_knowledge_graph(csv_path="database-all-triples.csv", 
                                   use_word_embeddings=True, 
                                   covid_similarity_threshold=0.6,
                                   ndd_similarity_threshold=0.6):
    """
    COVID-NDD enrichment analysis using semantic similarity for node identification
    """
    
    print("Loading complete knowledge graph from CSV...")
    
    # Load the complete triples dataset
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} triples from {csv_path}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst few rows:")
        print(df.head())
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    # Build complete network
    print("\nBuilding complete knowledge graph network...")
    G = build_network_from_triples(df)
    
    print(f"Network statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Connected components: {nx.number_connected_components(G)}")
    
    # Identify COVID and NDD nodes using semantic similarity
    all_nodes = list(G.nodes())
    
    print(f"\nUsing semantic similarity for node identification...")
    print(f"  COVID similarity threshold: {covid_similarity_threshold}")
    print(f"  NDD similarity threshold: {ndd_similarity_threshold}")
    covid_nodes, covid_scores = identify_covid_nodes_semantic(all_nodes, 
                                                            threshold=covid_similarity_threshold,
                                                            use_embeddings=use_word_embeddings)
    ndd_nodes, ndd_scores = identify_ndd_nodes_semantic(all_nodes, 
                                                        threshold=ndd_similarity_threshold,
                                                        use_embeddings=use_word_embeddings)

    
    print(f"\nSemantic node identification results:")
    print(f"  COVID nodes found: {len(covid_nodes)}")
    print(f"  NDD nodes found: {len(ndd_nodes)}")
    
    if len(covid_nodes) > 0:
        print(f"  Sample COVID nodes: {covid_nodes[:5]}")
    if len(ndd_nodes) > 0:
        print(f"  Sample NDD nodes: {ndd_nodes[:5]}")
    
    if len(covid_nodes) == 0 or len(ndd_nodes) == 0:
        print("ERROR: Insufficient COVID or NDD nodes found!")
        print("Try lowering similarity thresholds or check node naming conventions.")
        return None
    
    # Comprehensive enrichment analysis
    results = run_comprehensive_enrichment_analysis(G, covid_nodes, ndd_nodes, df)
    
    # Additional analysis of matched nodes
    print(f"\nPerforming additional node analysis...")
    analyze_node_categories(covid_nodes, ndd_nodes)
    
    # Optionally save detailed results to file
    save_node_matching_results(covid_nodes, covid_scores, ndd_nodes, ndd_scores, all_nodes)
    
    # Create comprehensive supplementary materials
    supplementary_dir = create_supplementary_materials(results, covid_nodes, covid_scores, 
                                                     ndd_nodes, ndd_scores, G, all_nodes)
    
    return results, supplementary_dir

def build_network_from_triples(df):
    """Build NetworkX graph from triples DataFrame"""
    
    G = nx.Graph()  # Use undirected graph for pathway analysis
    
    # Add edges from subject-object pairs
    for _, row in df.iterrows():
        subject = str(row['Subject']).strip()
        obj = str(row['Object']).strip()
        predicate = str(row['Predicate']).strip()
        
        if subject and obj and subject != 'nan' and obj != 'nan':
            # Add edge with predicate as attribute
            G.add_edge(subject, obj, predicate=predicate)
    
    return G

def identify_covid_nodes_semantic(all_nodes, threshold=0.3, use_embeddings=True):
    """
    Identify COVID-related nodes using semantic similarity
    """
    print("  Identifying COVID nodes with semantic similarity...")
    
    # Define comprehensive COVID seed terms
    covid_seed_terms = [
        "covid-19 coronavirus disease",
        "sars-cov-2 severe acute respiratory syndrome",
        "coronavirus pandemic virus",
        "covid pneumonia respiratory infection",
        "spike protein ace2 receptor",
        "coronavirus replication transcription",
        "viral entry host cell",
        "covid symptoms fever cough",
        "coronavirus transmission airborne",
        "sars outbreak epidemic"
    ]
    
    if use_embeddings:
        try:
            import spacy
            nlp = spacy.load("en_core_web_md")
            return _identify_nodes_spacy_embeddings(all_nodes, covid_seed_terms, nlp, threshold)
        except ImportError:
            print("    spaCy not available, using TF-IDF similarity")
            use_embeddings = False

    if not use_embeddings:
        return _identify_nodes_tfidf_similarity(all_nodes, covid_seed_terms, threshold)

def identify_ndd_nodes_semantic(all_nodes, threshold=0.3, use_embeddings=True):
    """
    Identify neurodegenerative disease nodes using semantic similarity
    """
    print("  Identifying NDD nodes with semantic similarity...")
    
    # Define comprehensive NDD seed terms
    ndd_seed_terms = [
        "alzheimer disease dementia memory loss",
        "parkinson disease movement disorder tremor",
        "neurodegeneration protein aggregation",
        "huntington disease genetic disorder",
        "amyotrophic lateral sclerosis motor neuron",
        "frontotemporal dementia behavioral changes",
        "lewy body disease alpha synuclein",
        "multiple sclerosis autoimmune neurologic",
        "prion disease spongiform encephalopathy",
        "tau protein neurofibrillary tangles",
        "beta amyloid plaques brain pathology",
        "dopamine neuron degeneration substantia nigra",
        "cognitive decline memory impairment",
        "motor neuron disease muscle weakness",
        "spinocerebellar ataxia coordination disorder"
    ]
    
    if use_embeddings:
        try:
            import spacy
            nlp = spacy.load("en_core_web_md")
            return _identify_nodes_spacy_embeddings(all_nodes, ndd_seed_terms, nlp, threshold)
        except ImportError:
            print("    spaCy not available, using TF-IDF similarity")
            use_embeddings = False

    if not use_embeddings:
        return _identify_nodes_tfidf_similarity(all_nodes, ndd_seed_terms, threshold)

def identify_covid_nodes_keyword(all_nodes):
    """
    Identify COVID-related nodes using simple keyword matching (for comparison)
    """
    covid_keywords = ['covid', 'coronavirus', 'sars-cov-2', 'sars', 'pandemic', 'viral']
    covid_nodes = []
    
    for node in all_nodes:
        node_lower = str(node).lower()
        if any(keyword in node_lower for keyword in covid_keywords):
            covid_nodes.append(node)
    
    return covid_nodes

def identify_ndd_nodes_keyword(all_nodes):
    """
    Identify NDD-related nodes using simple keyword matching (for comparison)
    """
    ndd_keywords = ['alzheimer', 'parkinson', 'dementia', 'neurodegeneration', 
                    'huntington', 'amyotrophic', 'multiple sclerosis', 'als']
    ndd_nodes = []
    
    for node in all_nodes:
        node_lower = str(node).lower()
        if any(keyword in node_lower for keyword in ndd_keywords):
            ndd_nodes.append(node)
    
    return ndd_nodes

def analyze_node_categories(covid_nodes, ndd_nodes):
    """
    Basic analysis of node categories found
    """
    print(f"Node Category Analysis:")
    print(f"  COVID nodes: {len(covid_nodes)}")
    print(f"  NDD nodes: {len(ndd_nodes)}")
    
    # Check for overlap
    covid_set = set(covid_nodes)
    ndd_set = set(ndd_nodes)
    overlap = covid_set.intersection(ndd_set)
    
    print(f"  Overlapping nodes: {len(overlap)}")
    if overlap:
        print(f"  Overlap examples: {list(overlap)[:5]}")
    
    # Basic categorization
    def count_by_type(nodes, type_name):
        protein_count = sum(1 for node in nodes if 'protein' in str(node).lower())
        gene_count = sum(1 for node in nodes if any(term in str(node).lower() 
                        for term in ['gene', 'dna', 'rna']))
        pathway_count = sum(1 for node in nodes if 'pathway' in str(node).lower())
        
        print(f"  {type_name} breakdown:")
        print(f"    Proteins: {protein_count}")
        print(f"    Genes: {gene_count}")
        print(f"    Pathways: {pathway_count}")
    
    count_by_type(covid_nodes, "COVID")
    count_by_type(ndd_nodes, "NDD")

def save_node_matching_results(covid_nodes, covid_scores, ndd_nodes, ndd_scores, all_nodes):
    """
    Save detailed node matching results to files
    """
    print("  Saving detailed node matching results...")
    
    try:
        # Save COVID nodes with scores
        covid_df = pd.DataFrame({
            'Node': covid_nodes,
            'Similarity_Score': covid_scores
        })
        covid_df = covid_df.sort_values('Similarity_Score', ascending=False)
        covid_df.to_csv('covid_nodes_semantic.csv', index=False)
        
        # Save NDD nodes with scores
        ndd_df = pd.DataFrame({
            'Node': ndd_nodes,
            'Similarity_Score': ndd_scores
        })
        ndd_df = ndd_df.sort_values('Similarity_Score', ascending=False)
        ndd_df.to_csv('ndd_nodes_semantic.csv', index=False)
        
        print(f"    ✅ Saved COVID nodes to: covid_nodes_semantic.csv")
        print(f"    ✅ Saved NDD nodes to: ndd_nodes_semantic.csv")
        
    except Exception as e:
        print(f"    ❌ Error saving node results: {e}")

def _identify_nodes_spacy_embeddings(all_nodes, seed_terms, nlp, threshold):
    """
    Use spaCy word embeddings for semantic similarity
    """
    print(f"    Using spaCy embeddings (threshold: {threshold})")
    
    # Get embeddings for seed terms
    seed_embeddings = []
    for term in seed_terms:
        doc = nlp(term)
        if doc.has_vector:
            # Average word vectors in the phrase
            embeddings = [token.vector for token in doc if token.has_vector and not token.is_stop]
            if embeddings:
                seed_embeddings.append(np.mean(embeddings, axis=0))
    
    if not seed_embeddings:
        print("    No embeddings found for seed terms")
        return []
    
    # Calculate centroid of seed embeddings
    seed_centroid = np.mean(seed_embeddings, axis=0)
    
    # Find similar nodes
    similar_nodes = []
    similarity_scores = []
    
    for node in all_nodes:
        # Clean and process node name
        node_text = _clean_node_text(str(node))
        node_doc = nlp(node_text)
        
        if node_doc.has_vector:
            # Get node embedding (average of word vectors)
            node_embeddings = [token.vector for token in node_doc 
                             if token.has_vector and not token.is_stop]
            
            if node_embeddings:
                node_vector = np.mean(node_embeddings, axis=0)
                
                # Calculate cosine similarity
                similarity = cosine_similarity([node_vector], [seed_centroid])[0][0]
                
                if similarity >= threshold:
                    similar_nodes.append(node)
                    similarity_scores.append(similarity)
    
    # Sort by similarity score
    node_score_pairs = list(zip(similar_nodes, similarity_scores))
    node_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"    Found {len(similar_nodes)} nodes above threshold")
    if node_score_pairs:
        print(f"    Top matches:")
        for node, score in node_score_pairs[:5]:
            print(f"      {node}: {score:.3f}")
    
    nodes = [node for node, score in node_score_pairs]
    scores = [score for node, score in node_score_pairs]
    return nodes, scores


def _identify_nodes_tfidf_similarity(all_nodes, seed_terms, threshold):
    """
    Use TF-IDF vectors for semantic similarity (fallback method)
    """
    print(f"    Using TF-IDF similarity (threshold: {threshold})")
    
    # Prepare text corpus
    all_node_texts = [_clean_node_text(str(node)) for node in all_nodes]
    
    # Add seed terms to corpus for comparison
    seed_text = " ".join(seed_terms)
    corpus = all_node_texts + [seed_text]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams for better context
        min_df=1,
        max_features=10000,
        lowercase=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Get similarity to seed terms (last item in corpus)
    seed_vector = tfidf_matrix[-1]
    node_vectors = tfidf_matrix[:-1]
    
    # Calculate cosine similarities
    similarities = cosine_similarity(node_vectors, seed_vector).flatten()
    
    # Find nodes above threshold
    similar_nodes = []
    similarity_scores = []
    
    for i, similarity in enumerate(similarities):
        if similarity >= threshold:
            similar_nodes.append(all_nodes[i])
            similarity_scores.append(similarity)
    
    # Sort by similarity score
    node_score_pairs = list(zip(similar_nodes, similarity_scores))
    node_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"    Found {len(similar_nodes)} nodes above threshold")
    if node_score_pairs:
        print(f"    Top matches:")
        for node, score in node_score_pairs[:5]:
            print(f"      {node}: {score:.3f}")
    
    nodes = [node for node, score in node_score_pairs]
    scores = [score for node, score in node_score_pairs]
    return nodes, scores


def _clean_node_text(node_text):
    """
    Clean and normalize node text for better semantic matching
    """
    # Convert to lowercase
    text = node_text.lower()
    
    # Replace underscores and hyphens with spaces
    text = re.sub(r'[-_]+', ' ', text)
    
    # Remove special characters but keep letters, numbers, and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ==========================================
# REST OF THE ANALYSIS CODE (UNCHANGED)
# ==========================================

def run_comprehensive_enrichment_analysis(G, covid_nodes, ndd_nodes, df_triples, n_iter=1000):
    """
    Run comprehensive enrichment analysis comparing multiple approaches
    """
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE COVID-NDD ENRICHMENT ANALYSIS")
    print(f"{'='*60}")
    
    results = {}
    
    # Analysis 1: Direct connectivity enrichment
    print("\n1. Testing direct connectivity enrichment...")
    direct_results = test_direct_connectivity_enrichment(G, covid_nodes, ndd_nodes, n_iter)
    results['direct_connectivity'] = direct_results
    
    # Analysis 2: Shortest path enrichment  
    print("\n2. Testing shortest path enrichment...")
    shortest_path_results = test_shortest_path_enrichment(G, covid_nodes, ndd_nodes, n_iter)
    results['shortest_paths'] = shortest_path_results
    
    # Analysis 3: Random walk proximity enrichment
    print("\n3. Testing random walk proximity...")
    proximity_results = test_proximity_enrichment(G, covid_nodes, ndd_nodes, n_iter)
    results['proximity'] = proximity_results
    
    # Analysis 4: Triple-level analysis
    print("\n4. Testing triple-level co-occurrence...")
    triple_results = test_triple_cooccurrence(df_triples, covid_nodes, ndd_nodes, n_iter)
    results['triple_cooccurrence'] = triple_results

    # Analysis 5: Degree Preserving Enrichment
    print("\n5. Testing degree-preserving enrichment...")
    degree_preserving_results = test_degree_preserving_enrichment(G, covid_nodes, ndd_nodes, n_iter)
    results['degree_preserving'] = degree_preserving_results
    
    # Statistical summary with multiple testing correction
    print_statistical_summary(results)
    
    return results

def test_direct_connectivity_enrichment(G, covid_nodes, ndd_nodes, n_iter=1000):
    """Test if COVID and NDD nodes are more directly connected than expected"""
    
    covid_set = set(covid_nodes)
    ndd_set = set(ndd_nodes)
    
    # Count observed direct connections
    observed_connections = 0
    for covid_node in covid_nodes:
        if covid_node in G:
            neighbors = set(G.neighbors(covid_node))
            observed_connections += len(neighbors.intersection(ndd_set))
    
    print(f"   Observed direct COVID-NDD connections: {observed_connections}")
    
    # Null distribution: random node sets of same sizes
    null_connections = []
    all_nodes = list(G.nodes())
    
    for _ in range(n_iter):
        # Sample random node sets of same sizes
        random_covid = random.sample(all_nodes, min(len(covid_nodes), len(all_nodes)))
        random_ndd = random.sample(all_nodes, min(len(ndd_nodes), len(all_nodes)))
        
        random_ndd_set = set(random_ndd)
        random_count = 0
        
        for node in random_covid:
            if node in G:
                neighbors = set(G.neighbors(node))
                random_count += len(neighbors.intersection(random_ndd_set))
        
        null_connections.append(random_count)
    
    return calculate_enrichment_statistics(observed_connections, null_connections, "direct connections")

def test_shortest_path_enrichment(G, covid_nodes, ndd_nodes, n_iter=1000, max_path_length=5):
    """Test if COVID-NDD shortest paths are shorter than expected"""
    
    # Find shortest paths between COVID and NDD nodes
    covid_ndd_paths = []
    path_lengths = []
    
    print(f"   Finding shortest paths (max length {max_path_length})...")
    
    for covid_node in random.sample(covid_nodes, min(50, len(covid_nodes))):#covid_nodes[:10]:  # Limit for computational efficiency
        for ndd_node in ndd_nodes[:10]:
            if covid_node in G and ndd_node in G and covid_node != ndd_node:
                try:
                    path = nx.shortest_path(G, covid_node, ndd_node)
                    if len(path) <= max_path_length:
                        covid_ndd_paths.append(path)
                        path_lengths.append(len(path))
                except nx.NetworkXNoPath:
                    continue
    
    if not path_lengths:
        print("   No paths found between COVID and NDD nodes!")
        return None
    
    observed_avg_length = np.mean(path_lengths)
    observed_path_count = len(covid_ndd_paths)
    
    print(f"   Found {observed_path_count} COVID-NDD paths")
    print(f"   Average path length: {observed_avg_length:.2f}")
    
    # Null distribution: random node pairs
    null_lengths = []
    null_counts = []
    all_nodes = list(G.nodes())
    
    for _ in range(n_iter):
        random_covid = random.sample(all_nodes, min(10, len(all_nodes)))
        random_ndd = random.sample(all_nodes, min(10, len(all_nodes)))
        
        random_paths = []
        for node1 in random_covid:
            for node2 in random_ndd:
                if node1 != node2:
                    try:
                        path = nx.shortest_path(G, node1, node2)
                        if len(path) <= max_path_length:
                            random_paths.append(len(path))
                    except nx.NetworkXNoPath:
                        continue
        
        if random_paths:
            null_lengths.append(np.mean(random_paths))
            null_counts.append(len(random_paths))
        else:
            null_lengths.append(float('inf'))
            null_counts.append(0)
    
    # Filter out infinite values
    null_lengths = [x for x in null_lengths if x != float('inf')]
    
    results = {
        'path_count': calculate_enrichment_statistics(observed_path_count, null_counts, "path count"),
        'avg_length': calculate_enrichment_statistics(observed_avg_length, null_lengths, "average path length")
    }
    
    return results

def test_proximity_enrichment(G, covid_nodes, ndd_nodes, n_iter=1000):
    """Test proximity between COVID and NDD node sets using random walks"""
    
    # Calculate average shortest distance between COVID and NDD nodes
    distances = []
    
    print("   Calculating node proximities...")
    
    for covid_node in random.sample(covid_nodes, min(20, len(covid_nodes))): #covid_nodes[:5]:  # Limit for efficiency
        for ndd_node in ndd_nodes[:5]:
            if covid_node in G and ndd_node in G and covid_node != ndd_node:
                try:
                    dist = nx.shortest_path_length(G, covid_node, ndd_node)
                    distances.append(dist)
                except nx.NetworkXNoPath:
                    continue
    
    if not distances:
        return None
    
    observed_avg_distance = np.mean(distances)
    print(f"   Average COVID-NDD distance: {observed_avg_distance:.2f}")
    
    # Null distribution
    null_distances = []
    all_nodes = list(G.nodes())
    
    for _ in range(n_iter):
        random_set1 = random.sample(all_nodes, min(5, len(all_nodes)))
        random_set2 = random.sample(all_nodes, min(5, len(all_nodes)))
        
        random_dists = []
        for node1 in random_set1:
            for node2 in random_set2:
                if node1 != node2:
                    try:
                        dist = nx.shortest_path_length(G, node1, node2)
                        random_dists.append(dist)
                    except nx.NetworkXNoPath:
                        continue
        
        if random_dists:
            null_distances.append(np.mean(random_dists))
    
    return calculate_enrichment_statistics(observed_avg_distance, null_distances, "average distance", 
                                         lower_is_better=True)

def test_triple_cooccurrence(df_triples, covid_nodes, ndd_nodes, n_iter=1000):
    """Test if COVID and NDD terms co-occur in triples more than expected"""
    
    covid_set = set(covid_nodes)
    ndd_set = set(ndd_nodes)
    
    # Count triples containing both COVID and NDD terms
    observed_cooccurrence = 0
    total_triples = len(df_triples)
    
    for _, row in df_triples.iterrows():
        subject = str(row['Subject'])
        obj = str(row['Object'])
        
        has_covid = subject in covid_set or obj in covid_set
        has_ndd = subject in ndd_set or obj in ndd_set
        
        if has_covid and has_ndd:
            observed_cooccurrence += 1
    
    print(f"   Observed COVID-NDD co-occurring triples: {observed_cooccurrence}/{total_triples}")
    print(f"   Observed rate: {100*observed_cooccurrence/total_triples:.2f}%")
    
    # Null distribution: random node label assignment
    null_cooccurrences = []
    all_unique_nodes = set(df_triples['Subject']).union(set(df_triples['Object']))
    all_unique_nodes = list(all_unique_nodes)
    
    for _ in range(n_iter):
        # Randomly assign COVID and NDD labels
        random_covid = set(random.sample(all_unique_nodes, len(covid_nodes)))
        random_ndd = set(random.sample(all_unique_nodes, len(ndd_nodes)))
        
        random_count = 0
        for _, row in df_triples.iterrows():
            subject = str(row['Subject'])
            obj = str(row['Object'])
            
            has_covid = subject in random_covid or obj in random_covid
            has_ndd = subject in random_ndd or obj in random_ndd
            
            if has_covid and has_ndd:
                random_count += 1
        
        null_cooccurrences.append(random_count)
    
    return calculate_enrichment_statistics(observed_cooccurrence, null_cooccurrences, "co-occurring triples")

def calculate_enrichment_statistics(observed, null_distribution, metric_name, lower_is_better=False):
    """Calculate comprehensive enrichment statistics"""
    
    null_array = np.array(null_distribution)
    mean_null = np.mean(null_array)
    std_null = np.std(null_array)
    
    # P-value calculation
    if lower_is_better:
        # For metrics where lower values indicate enrichment (e.g., distance)
        p_value = (np.sum(null_array <= observed) + 1) / (len(null_array) + 1)
        enrichment = mean_null / observed if observed > 0 else float('inf')
    else:
        # For metrics where higher values indicate enrichment
        p_value = (np.sum(null_array >= observed) + 1) / (len(null_array) + 1)
        enrichment = observed / mean_null if mean_null > 0 else float('inf')
    
    # Z-score
    z_score = (observed - mean_null) / std_null if std_null > 0 else float('inf')
    
    # Effect size (Cohen's d)
    cohens_d = abs(observed - mean_null) / std_null if std_null > 0 else float('inf')
    
    return {
        'metric': metric_name,
        'observed': observed,
        'null_mean': mean_null,
        'null_std': std_null,
        'p_value': p_value,
        'enrichment': enrichment,
        'z_score': z_score,
        'cohens_d': cohens_d,
        'lower_is_better': lower_is_better
    }

def test_degree_preserving_enrichment(G, covid_nodes, ndd_nodes, n_iter=1000):
    """
    Test COVID-NDD connectivity against degree-preserving random graphs.
    """
    
    covid_set = set(covid_nodes)
    ndd_set = set(ndd_nodes)

    # Compute observed number of direct COVID–NDD connections
    observed_connections = 0
    for covid_node in covid_nodes:
        if covid_node in G:
            neighbors = set(G.neighbors(covid_node))
            observed_connections += len(neighbors.intersection(ndd_set))

    print(f"   Observed COVID–NDD connections: {observed_connections}")

    null_connections = []

    for i in range(n_iter):
        # Create a copy of G and rewire it while preserving degrees
        G_random = G.copy()
        nx.double_edge_swap(G_random, nswap=20*G.number_of_edges(), max_tries=100*G.number_of_edges())

        random_count = 0
        for covid_node in covid_nodes:
            if covid_node in G_random:
                neighbors = set(G_random.neighbors(covid_node))
                random_count += len(neighbors.intersection(ndd_set))

        null_connections.append(random_count)

        if (i+1) % 100 == 0:
            print(f"   Iteration {i+1}/{n_iter} complete")

    result = calculate_enrichment_statistics(observed_connections, null_connections, 
                                             "direct COVID–NDD connections (degree-preserving)")

    return result

def print_statistical_summary(results):
    """Print comprehensive statistical summary with multiple testing correction"""
    
    print(f"\n{'='*80}")
    print("STATISTICAL SUMMARY WITH MULTIPLE TESTING CORRECTION")
    print(f"{'='*80}")
    
    # Collect all p-values for correction
    all_p_values = []
    test_names = []
    all_results = []
    
    for analysis_type, analysis_results in results.items():
        if analysis_results is None:
            continue
            
        if isinstance(analysis_results, dict) and 'p_value' in analysis_results:
            # Single test result
            all_p_values.append(analysis_results['p_value'])
            test_names.append(analysis_type)
            all_results.append(analysis_results)
        elif isinstance(analysis_results, dict):
            # Multiple sub-tests
            for sub_test, sub_result in analysis_results.items():
                if isinstance(sub_result, dict) and 'p_value' in sub_result:
                    all_p_values.append(sub_result['p_value'])
                    test_names.append(f"{analysis_type}_{sub_test}")
                    all_results.append(sub_result)
    
    if not all_p_values:
        print("No valid statistical tests found!")
        return
    
    # Apply multiple testing corrections
    bonferroni_corrected = multipletests(all_p_values, method='bonferroni')[1]
    fdr_corrected = multipletests(all_p_values, method='fdr_bh')[1]
    
    # Print detailed results
    for i, (test_name, result) in enumerate(zip(test_names, all_results)):
        print(f"\n{test_name.upper().replace('_', ' ')}:")
        print(f"  Metric: {result['metric']}")
        print(f"  Observed: {result['observed']:.4f}")
        print(f"  Expected (null): {result['null_mean']:.4f} ± {result['null_std']:.4f}")
        print(f"  Raw p-value: {result['p_value']:.4f}")
        print(f"  Bonferroni p-value: {bonferroni_corrected[i]:.4f}")
        print(f"  FDR p-value: {fdr_corrected[i]:.4f}")
        print(f"  Enrichment: {result['enrichment']:.2f}x")
        print(f"  Z-score: {result['z_score']:.2f}")
        print(f"  Effect size (Cohen's d): {result['cohens_d']:.2f}")
        
        # Add interpretation for distance metrics
        if result.get('lower_is_better', False):
            print(f"  (Lower values indicate stronger connectivity)")
        
        # Significance indicators
        if bonferroni_corrected[i] < 0.05:
            print(f"  *** SIGNIFICANT after Bonferroni correction ***")
        elif fdr_corrected[i] < 0.05:
            print(f"  ** SIGNIFICANT after FDR correction **")
        elif result['p_value'] < 0.05:
            print(f"  * Nominally significant (uncorrected) *")
        else:
            print(f"  Not significant")
    
    # Overall summary
    n_significant_bonf = sum(p < 0.05 for p in bonferroni_corrected)
    n_significant_fdr = sum(p < 0.05 for p in fdr_corrected)
    n_nominal = sum(p < 0.05 for p in all_p_values)
    
    print(f"\n{'='*50}")
    print(f"OVERALL SIGNIFICANCE SUMMARY:")
    print(f"  Total tests performed: {len(all_p_values)}")
    print(f"  Nominally significant (p < 0.05): {n_nominal}")
    print(f"  Bonferroni significant (p < 0.05): {n_significant_bonf}")
    print(f"  FDR significant (p < 0.05): {n_significant_fdr}")
    
    print(f"\n{'='*50}")
    print(f"SEMANTIC SIMILARITY INSIGHTS:")
    print(f"  Method: Enhanced semantic matching vs. keyword matching")
    print(f"  Captures: Conceptually related terms, synonyms, variations")
    print(f"  Benefits: More comprehensive node identification")
    print(f"{'='*50}")

def run_semantic_analysis_with_thresholds(csv_path="database-all-triples.csv"):
    """
    Run analysis with multiple threshold settings to find optimal parameters
    """
    print("="*60)
    print("SEMANTIC SIMILARITY THRESHOLD OPTIMIZATION")
    print("="*60)
    
    # Test different threshold combinations
    threshold_combinations = [
        (0.2, 0.2),  # Very inclusive
        (0.3, 0.3),  # Moderately inclusive  
        (0.4, 0.4),  # Moderately strict
        (0.5, 0.5)   # Very strict
    ]
    
    for covid_thresh, ndd_thresh in threshold_combinations:
        print(f"\nTesting thresholds: COVID={covid_thresh}, NDD={ndd_thresh}")
        
        results = analyze_complete_knowledge_graph(
            csv_path=csv_path,
            covid_similarity_threshold=covid_thresh,
            ndd_similarity_threshold=ndd_thresh
        )
        
        if results:
            print(f"✓ Analysis completed with thresholds COVID={covid_thresh}, NDD={ndd_thresh}")
        else:
            print(f"✗ Analysis failed with thresholds COVID={covid_thresh}, NDD={ndd_thresh}")

# Draw visualization of tests

def create_supplementary_materials(results, covid_nodes, covid_scores, ndd_nodes, ndd_scores, 
                                 G, all_nodes, output_dir="supplementary_materials"):
    """
    Create comprehensive supplementary materials for publication
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("CREATING SUPPLEMENTARY MATERIALS")
    print(f"{'='*60}")
    
    # 1. Statistical Results Summary Table
    create_statistical_summary_table(results, output_dir)
    
    # 2. Node Identification Comparison Table
    create_node_comparison_table(covid_nodes, covid_scores, ndd_nodes, ndd_scores, 
                               all_nodes, output_dir)
    
    # 3. Network Connectivity Visualization
    create_network_visualization(G, covid_nodes, ndd_nodes, output_dir)
    
    # 4. Enrichment Analysis Bar Chart
    create_enrichment_bar_chart(results, output_dir)
    
    # 5. Node Categories Analysis
    create_node_categories_analysis(covid_nodes, ndd_nodes, output_dir)
    
    # 6. Connectivity Heatmap
    create_connectivity_heatmap(G, covid_nodes, ndd_nodes, output_dir)
    
    # 7. P-value Distribution Plot
    create_pvalue_distribution_plot(results, output_dir)
    
    # 8. Network Statistics Table
    create_network_statistics_table(G, covid_nodes, ndd_nodes, output_dir)
    
    print(f"\n✅ All supplementary materials created in: {output_dir}/")
    return output_dir

def create_statistical_summary_table(results, output_dir):
    """
    Create publication-ready statistical results table
    """
    print("  Creating statistical summary table...")
    
    # Collect all results for table
    table_data = []
    
    for analysis_type, analysis_results in results.items():
        if analysis_results is None:
            continue
            
        if isinstance(analysis_results, dict) and 'p_value' in analysis_results:
            # Single test result
            table_data.append({
                'Analysis': analysis_type.replace('_', ' ').title(),
                'Metric': analysis_results['metric'],
                'Observed': analysis_results['observed'],
                'Expected': analysis_results['null_mean'],
                'Std_Dev': analysis_results['null_std'],
                'Raw_P_Value': analysis_results['p_value'],
                'Enrichment': analysis_results['enrichment'],
                'Z_Score': analysis_results['z_score'],
                'Cohens_D': analysis_results['cohens_d']
            })
        elif isinstance(analysis_results, dict):
            # Multiple sub-tests
            for sub_test, sub_result in analysis_results.items():
                if isinstance(sub_result, dict) and 'p_value' in sub_result:
                    table_data.append({
                        'Analysis': f"{analysis_type.replace('_', ' ').title()} - {sub_test.replace('_', ' ').title()}",
                        'Metric': sub_result['metric'],
                        'Observed': sub_result['observed'],
                        'Expected': sub_result['null_mean'],
                        'Std_Dev': sub_result['null_std'],
                        'Raw_P_Value': sub_result['p_value'],
                        'Enrichment': sub_result['enrichment'],
                        'Z_Score': sub_result['z_score'],
                        'Cohens_D': sub_result['cohens_d']
                    })
    
    if not table_data:
        return
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Apply multiple testing corrections
    raw_p_values = df['Raw_P_Value'].values
    bonferroni_corrected = multipletests(raw_p_values, method='bonferroni')[1]
    fdr_corrected = multipletests(raw_p_values, method='fdr_bh')[1]
    
    df['Bonferroni_P'] = bonferroni_corrected
    df['FDR_P'] = fdr_corrected
    
    # Add significance indicators
    df['Significance'] = 'Not Significant'
    df.loc[df['Bonferroni_P'] < 0.05, 'Significance'] = 'Bonferroni Significant'
    df.loc[(df['FDR_P'] < 0.05) & (df['Bonferroni_P'] >= 0.05), 'Significance'] = 'FDR Significant'
    df.loc[(df['Raw_P_Value'] < 0.05) & (df['FDR_P'] >= 0.05), 'Significance'] = 'Nominally Significant'
    
    # Round numerical columns
    numerical_cols = ['Observed', 'Expected', 'Std_Dev', 'Raw_P_Value', 'Bonferroni_P', 
                     'FDR_P', 'Enrichment', 'Z_Score', 'Cohens_D']
    for col in numerical_cols:
        if col in ['Raw_P_Value', 'Bonferroni_P', 'FDR_P']:
            df[col] = df[col].round(4)
        else:
            df[col] = df[col].round(3)
    
    # Save table
    df.to_csv(f"{output_dir}/Table_S1_Statistical_Results.csv", index=False)
    
    # Create LaTeX version
    latex_table = df.to_latex(index=False, float_format='{:.3f}'.format, 
                             caption="Statistical analysis results for COVID-NDD connectivity enrichment tests",
                             label="tab:statistical_results")
    
    with open(f"{output_dir}/Table_S1_Statistical_Results.tex", 'w') as f:
        f.write(latex_table)
    
    print(f"    ✅ Statistical table saved: Table_S1_Statistical_Results.csv/.tex")

def create_node_comparison_table(covid_nodes, covid_scores, ndd_nodes, ndd_scores, 
                               all_nodes, output_dir):
    """
    Create table comparing keyword vs semantic node identification
    """
    print("  Creating node identification comparison table...")
    
    # Get keyword-based results for comparison
    keyword_covid = identify_covid_nodes_keyword(all_nodes)
    keyword_ndd = identify_ndd_nodes_keyword(all_nodes)
    
    # Create comparison data
    comparison_data = {
        'Method': ['Keyword Matching', 'Semantic Similarity', 'Improvement'],
        'COVID_Nodes': [len(keyword_covid), len(covid_nodes), len(covid_nodes) - len(keyword_covid)],
        'NDD_Nodes': [len(keyword_ndd), len(ndd_nodes), len(ndd_nodes) - len(keyword_ndd)],
        'COVID_Coverage_Pct': [
            round(100 * len(keyword_covid) / len(all_nodes), 2),
            round(100 * len(covid_nodes) / len(all_nodes), 2),
            round(100 * (len(covid_nodes) - len(keyword_covid)) / len(all_nodes), 2)
        ],
        'NDD_Coverage_Pct': [
            round(100 * len(keyword_ndd) / len(all_nodes), 2),
            round(100 * len(ndd_nodes) / len(all_nodes), 2),
            round(100 * (len(ndd_nodes) - len(keyword_ndd)) / len(all_nodes), 2)
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(f"{output_dir}/Table_S2_Node_Identification_Comparison.csv", index=False)
    
    # Create detailed node lists
    covid_detailed = pd.DataFrame({
        'Node': covid_nodes,
        'Similarity_Score': covid_scores,
        'Method': ['Semantic'] * len(covid_nodes)
    })
    
    # Mark overlap with keyword method
    keyword_covid_set = set(keyword_covid)
    covid_detailed.loc[covid_detailed['Node'].isin(keyword_covid_set), 'Method'] = 'Both'
    
    ndd_detailed = pd.DataFrame({
        'Node': ndd_nodes,
        'Similarity_Score': ndd_scores,
        'Method': ['Semantic'] * len(ndd_nodes)
    })
    
    keyword_ndd_set = set(keyword_ndd)
    ndd_detailed.loc[ndd_detailed['Node'].isin(keyword_ndd_set), 'Method'] = 'Both'
    
    # Sort by similarity score
    covid_detailed = covid_detailed.sort_values('Similarity_Score', ascending=False)
    ndd_detailed = ndd_detailed.sort_values('Similarity_Score', ascending=False)
    
    # Save detailed lists
    covid_detailed.to_csv(f"{output_dir}/Table_S3_COVID_Nodes_Detailed.csv", index=False)
    ndd_detailed.to_csv(f"{output_dir}/Table_S4_NDD_Nodes_Detailed.csv", index=False)
    
    print(f"    ✅ Node comparison tables saved: Table_S2-S4")

def create_network_visualization(G, covid_nodes, ndd_nodes, output_dir, sample_size=100):
    """
    Create network visualization showing COVID-NDD connections
    """
    print("  Creating network visualization...")
    
    try:
        # Sample nodes for visualization if network is large
        if len(G.nodes()) > sample_size:
            # Get largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G_viz = G.subgraph(largest_cc)
            
            # Sample from the largest component
            nodes_to_include = set()
            
            # Include all COVID and NDD nodes that are in the component
            covid_in_cc = [n for n in covid_nodes if n in G_viz]
            ndd_in_cc = [n for n in ndd_nodes if n in G_viz]
            nodes_to_include.update(covid_in_cc)
            nodes_to_include.update(ndd_in_cc)
            
            # Add their immediate neighbors
            for node in list(nodes_to_include):
                if node in G_viz:
                    nodes_to_include.update(G_viz.neighbors(node))
            
            # Sample additional nodes to reach sample_size
            remaining_nodes = list(set(G_viz.nodes()) - nodes_to_include)
            if len(nodes_to_include) < sample_size and remaining_nodes:
                additional_needed = min(sample_size - len(nodes_to_include), len(remaining_nodes))
                nodes_to_include.update(random.sample(remaining_nodes, additional_needed))
            
            G_viz = G_viz.subgraph(list(nodes_to_include))
        else:
            G_viz = G
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Filter nodes that are actually in the visualization graph
        covid_in_viz = [n for n in covid_nodes if n in G_viz]
        ndd_in_viz = [n for n in ndd_nodes if n in G_viz]
        other_nodes = [n for n in G_viz.nodes() if n not in covid_in_viz and n not in ndd_in_viz]
        
        # Layout
        pos = nx.spring_layout(G_viz, k=1, iterations=50, seed=42)
        
        # Draw edges
        nx.draw_networkx_edges(G_viz, pos, alpha=0.3, width=0.5, edge_color='lightgray')
        
        # Highlight COVID-NDD edges
        covid_ndd_edges = []
        for covid_node in covid_in_viz:
            for ndd_node in ndd_in_viz:
                if G_viz.has_edge(covid_node, ndd_node):
                    covid_ndd_edges.append((covid_node, ndd_node))
        
        if covid_ndd_edges:
            nx.draw_networkx_edges(G_viz, pos, edgelist=covid_ndd_edges, 
                                 edge_color='red', width=2, alpha=0.8)
        
        # Draw nodes
        nx.draw_networkx_nodes(G_viz, pos, nodelist=other_nodes, 
                             node_color='lightblue', node_size=20, alpha=0.6)
        nx.draw_networkx_nodes(G_viz, pos, nodelist=covid_in_viz, 
                             node_color='red', node_size=100, alpha=0.8)
        nx.draw_networkx_nodes(G_viz, pos, nodelist=ndd_in_viz, 
                             node_color='blue', node_size=100, alpha=0.8)
        
        # Add labels for disease nodes only
        disease_nodes = covid_in_viz + ndd_in_viz
        disease_labels = {node: str(node)[:15] + '...' if len(str(node)) > 15 else str(node) 
                         for node in disease_nodes}
        
        nx.draw_networkx_labels(G_viz, pos, labels=disease_labels, 
                              font_size=8, font_weight='bold')
        
        # Legend
        legend_elements = [
            Patch(facecolor='red', label=f'COVID nodes (n={len(covid_in_viz)})'),
            Patch(facecolor='blue', label=f'NDD nodes (n={len(ndd_in_viz)})'),
            Patch(facecolor='lightblue', label=f'Other nodes (n={len(other_nodes)})'),
            Patch(facecolor='red', label=f'COVID-NDD edges (n={len(covid_ndd_edges)})')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f'COVID-NDD Network Connectivity\n({len(G_viz.nodes())} nodes, {len(G_viz.edges())} edges)', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/Figure_S1_Network_Visualization.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/Figure_S1_Network_Visualization.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Network visualization saved: Figure_S1_Network_Visualization.png/.pdf")
        
    except Exception as e:
        print(f"    ❌ Error creating network visualization: {e}")

def create_enrichment_bar_chart(results, output_dir):
    """
    Create bar chart showing enrichment results across different methods
    """
    print("  Creating enrichment analysis bar chart...")
    
    # Extract enrichment data
    enrichment_data = []
    p_values = []
    
    for analysis_type, analysis_results in results.items():
        if analysis_results is None:
            continue
            
        if isinstance(analysis_results, dict) and 'enrichment' in analysis_results:
            enrichment_data.append({
                'Method': analysis_type.replace('_', ' ').title(),
                'Enrichment': analysis_results['enrichment'],
                'P_Value': analysis_results['p_value'],
                'Significant': analysis_results['p_value'] < 0.05
            })
        elif isinstance(analysis_results, dict):
            for sub_test, sub_result in analysis_results.items():
                if isinstance(sub_result, dict) and 'enrichment' in sub_result:
                    enrichment_data.append({
                        'Method': f"{analysis_type.replace('_', ' ').title()}\n({sub_test.replace('_', ' ')})",
                        'Enrichment': sub_result['enrichment'],
                        'P_Value': sub_result['p_value'],
                        'Significant': sub_result['p_value'] < 0.05
                    })
    
    if not enrichment_data:
        return
    
    df_enrichment = pd.DataFrame(enrichment_data)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    
    # Color bars by significance
    colors = ['red' if sig else 'lightcoral' for sig in df_enrichment['Significant']]
    
    bars = plt.bar(range(len(df_enrichment)), df_enrichment['Enrichment'], 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add horizontal line at enrichment = 1 (no enrichment)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No enrichment')
    
    # Customize plot
    plt.xlabel('Analysis Method', fontsize=12, fontweight='bold')
    plt.ylabel('Enrichment Ratio\n(Observed/Expected)', fontsize=12, fontweight='bold')
    plt.title('COVID-NDD Connectivity Enrichment Analysis Results', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(df_enrichment)), df_enrichment['Method'], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, enrichment, p_val) in enumerate(zip(bars, df_enrichment['Enrichment'], df_enrichment['P_Value'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{enrichment:.2f}\np={p_val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Significant (p < 0.05)'),
        Patch(facecolor='lightcoral', alpha=0.7, label='Not significant')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/Figure_S2_Enrichment_Analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/Figure_S2_Enrichment_Analysis.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"    ✅ Enrichment bar chart saved: Figure_S2_Enrichment_Analysis.png/.pdf")

def create_node_categories_analysis(covid_nodes, ndd_nodes, output_dir):
    """
    Create visualization of node categories
    """
    print("  Creating node categories analysis...")
    
    # Define categories
    categories = {
        'Proteins': ['protein', 'enzyme', 'kinase', 'receptor', 'antibody', 'antigen'],
        'Genes': ['gene', 'dna', 'rna', 'genomic', 'genetic', 'allele'],
        'Pathways': ['pathway', 'signaling', 'cascade', 'regulation', 'metabolism'],
        'Diseases': ['disease', 'disorder', 'syndrome', 'infection', 'illness'],
        'Symptoms': ['symptom', 'fever', 'cough', 'pain', 'inflammation', 'cognitive'],
        'Drugs': ['drug', 'inhibitor', 'treatment', 'therapy', 'medication'],
        'Cellular': ['cell', 'cellular', 'membrane', 'organelle', 'nucleus'],
        'Immune': ['immune', 'immunological', 'cytokine', 'interferon', 'antibody']
    }
    
    def categorize_nodes(nodes):
        category_counts = {cat: 0 for cat in categories}
        uncategorized = 0
        
        for node in nodes:
            node_lower = str(node).lower()
            categorized = False
            
            for category, keywords in categories.items():
                if any(keyword in node_lower for keyword in keywords):
                    category_counts[category] += 1
                    categorized = True
                    break
            
            if not categorized:
                uncategorized += 1
        
        category_counts['Uncategorized'] = uncategorized
        return category_counts
    
    covid_categories = categorize_nodes(covid_nodes)
    ndd_categories = categorize_nodes(ndd_nodes)
    
    # Create comparison DataFrame
    categories_df = pd.DataFrame({
        'Category': list(covid_categories.keys()),
        'COVID_Count': list(covid_categories.values()),
        'NDD_Count': list(ndd_categories.values())
    })
    
    # Calculate percentages
    categories_df['COVID_Percent'] = (categories_df['COVID_Count'] / len(covid_nodes) * 100).round(1)
    categories_df['NDD_Percent'] = (categories_df['NDD_Count'] / len(ndd_nodes) * 100).round(1)
    
    # Save table
    categories_df.to_csv(f"{output_dir}/Table_S5_Node_Categories.csv", index=False)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # COVID pie chart
    covid_data = categories_df[categories_df['COVID_Count'] > 0]
    ax1.pie(covid_data['COVID_Count'], labels=covid_data['Category'], autopct='%1.1f%%', 
           startangle=90, colors=plt.cm.Set3(range(len(covid_data))))
    ax1.set_title(f'COVID Node Categories\n(Total: {len(covid_nodes)} nodes)', fontweight='bold')
    
    # NDD pie chart
    ndd_data = categories_df[categories_df['NDD_Count'] > 0]
    ax2.pie(ndd_data['NDD_Count'], labels=ndd_data['Category'], autopct='%1.1f%%', 
           startangle=90, colors=plt.cm.Set3(range(len(ndd_data))))
    ax2.set_title(f'NDD Node Categories\n(Total: {len(ndd_nodes)} nodes)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure_S3_Node_Categories.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/Figure_S3_Node_Categories.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"    ✅ Node categories analysis saved: Table_S5 and Figure_S3")

def create_connectivity_heatmap(G, covid_nodes, ndd_nodes, output_dir, max_nodes=50):
    """
    Create heatmap showing connectivity patterns between top COVID and NDD nodes
    """
    print("  Creating connectivity heatmap...")
    
    try:
        # Sample top nodes if there are too many
        covid_sample = covid_nodes[:max_nodes//2] if len(covid_nodes) > max_nodes//2 else covid_nodes
        ndd_sample = ndd_nodes[:max_nodes//2] if len(ndd_nodes) > max_nodes//2 else ndd_nodes
        
        # Create connectivity matrix
        connectivity_matrix = np.zeros((len(covid_sample), len(ndd_sample)))
        
        for i, covid_node in enumerate(covid_sample):
            for j, ndd_node in enumerate(ndd_sample):
                if covid_node in G and ndd_node in G:
                    if G.has_edge(covid_node, ndd_node):
                        connectivity_matrix[i, j] = 1
                    else:
                        # Check for paths of length 2-3
                        try:
                            path_length = nx.shortest_path_length(G, covid_node, ndd_node)
                            if path_length <= 3:
                                connectivity_matrix[i, j] = 1.0 / path_length
                        except nx.NetworkXNoPath:
                            connectivity_matrix[i, j] = 0
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Truncate long node names for display
        covid_labels = [str(node)[:20] + '...' if len(str(node)) > 20 else str(node) 
                       for node in covid_sample]
        ndd_labels = [str(node)[:20] + '...' if len(str(node)) > 20 else str(node) 
                     for node in ndd_sample]
        
        sns.heatmap(connectivity_matrix, 
                   xticklabels=ndd_labels, 
                   yticklabels=covid_labels,
                   cmap='Reds', 
                   cbar_kws={'label': 'Connectivity Strength'},
                   square=False)
        
        plt.title('COVID-NDD Node Connectivity Heatmap\n(1.0=direct, 0.5=2-hop, 0.33=3-hop)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('NDD Nodes', fontsize=12, fontweight='bold')
        plt.ylabel('COVID Nodes', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Figure_S4_Connectivity_Heatmap.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/Figure_S4_Connectivity_Heatmap.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Connectivity heatmap saved: Figure_S4_Connectivity_Heatmap.png/.pdf")
        
    except Exception as e:
        print(f"    ❌ Error creating connectivity heatmap: {e}")

def create_pvalue_distribution_plot(results, output_dir):
    """
    Create p-value distribution plot
    """
    print("  Creating p-value distribution plot...")
    
    # Extract p-values
    p_values = []
    test_names = []
    
    for analysis_type, analysis_results in results.items():
        if analysis_results is None:
            continue
            
        if isinstance(analysis_results, dict) and 'p_value' in analysis_results:
            p_values.append(analysis_results['p_value'])
            test_names.append(analysis_type.replace('_', ' ').title())
        elif isinstance(analysis_results, dict):
            for sub_test, sub_result in analysis_results.items():
                if isinstance(sub_result, dict) and 'p_value' in sub_result:
                    p_values.append(sub_result['p_value'])
                    test_names.append(f"{analysis_type.replace('_', ' ').title()} - {sub_test.replace('_', ' ').title()}")
    
    if not p_values:
        return
    
    # Apply multiple testing corrections
    bonferroni_corrected = multipletests(p_values, method='bonferroni')[1]
    fdr_corrected = multipletests(p_values, method='fdr_bh')[1]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Raw p-values
    colors = ['red' if p < 0.05 else 'lightcoral' for p in p_values]
    bars1 = ax1.bar(range(len(p_values)), p_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    ax1.set_ylabel('Raw P-value', fontweight='bold')
    ax1.set_title('Statistical Significance of COVID-NDD Connectivity Tests', fontweight='bold')
    ax1.set_xticks(range(len(test_names)))
    ax1.set_xticklabels(test_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Corrected p-values
    colors_bonf = ['red' if p < 0.05 else 'lightcoral' for p in bonferroni_corrected]
    colors_fdr = ['blue' if p < 0.05 else 'lightblue' for p in fdr_corrected]
    
    x = np.arange(len(p_values))
    width = 0.35
    
    ax2.bar(x - width/2, bonferroni_corrected, width, color=colors_bonf, alpha=0.7, 
           label='Bonferroni', edgecolor='black')
    ax2.bar(x + width/2, fdr_corrected, width, color=colors_fdr, alpha=0.7, 
           label='FDR', edgecolor='black')
    
    ax2.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='α = 0.05')
    ax2.set_ylabel('Corrected P-value', fontweight='bold')
    ax2.set_xlabel('Statistical Test', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure_S5_Pvalue_Distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/Figure_S5_Pvalue_Distribution.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"    ✅ P-value distribution plot saved: Figure_S5_Pvalue_Distribution.png/.pdf")

def create_network_statistics_table(G, covid_nodes, ndd_nodes, output_dir):
    """
    Create comprehensive network statistics table
    """
    print("  Creating network statistics table...")
    
    # Basic network statistics
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G)
    
    # Largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc_size = len(largest_cc)
    largest_cc_frac = largest_cc_size / n_nodes
    
    # Disease node statistics
    covid_in_network = len([n for n in covid_nodes if n in G])
    ndd_in_network = len([n for n in ndd_nodes if n in G])
    
    # Degree statistics
    degrees = dict(G.degree())
    covid_degrees = [degrees.get(n, 0) for n in covid_nodes if n in G]
    ndd_degrees = [degrees.get(n, 0) for n in ndd_nodes if n in G]
    all_degrees = list(degrees.values())
    
    # Direct connections
    direct_connections = 0
    for covid_node in covid_nodes:
        if covid_node in G:
            neighbors = set(G.neighbors(covid_node))
            direct_connections += len(neighbors.intersection(set(ndd_nodes)))
    
    # Create statistics table
    stats_data = {
        'Statistic': [
            'Total Nodes', 'Total Edges', 'Connected Components',
            'Largest Component Size', 'Largest Component Fraction',
            'COVID Nodes Total', 'COVID Nodes in Network', 'COVID Coverage',
            'NDD Nodes Total', 'NDD Nodes in Network', 'NDD Coverage',
            'Direct COVID-NDD Connections',
            'Average Degree (All)', 'Average Degree (COVID)', 'Average Degree (NDD)',
            'Max Degree (All)', 'Max Degree (COVID)', 'Max Degree (NDD)',
            'Network Density', 'Average Clustering Coefficient'
        ],
        'Value': [
            n_nodes, n_edges, n_components,
            largest_cc_size, f"{largest_cc_frac:.3f}",
            len(covid_nodes), covid_in_network, f"{covid_in_network/len(covid_nodes):.3f}",
            len(ndd_nodes), ndd_in_network, f"{ndd_in_network/len(ndd_nodes):.3f}",
            direct_connections,
            f"{np.mean(all_degrees):.2f}", 
            f"{np.mean(covid_degrees):.2f}" if covid_degrees else "N/A",
            f"{np.mean(ndd_degrees):.2f}" if ndd_degrees else "N/A",
            max(all_degrees), 
            max(covid_degrees) if covid_degrees else "N/A",
            max(ndd_degrees) if ndd_degrees else "N/A",
            f"{nx.density(G):.6f}",
            f"{nx.average_clustering(G):.3f}"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(f"{output_dir}/Table_S6_Network_Statistics.csv", index=False)
    
    print(f"    ✅ Network statistics table saved: Table_S6_Network_Statistics.csv")

if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("COVID-NDD Knowledge Graph Enrichment Analysis")
    print("Enhanced with Semantic Similarity")
    print("=" * 50)
    
    # Option 1: Run full analysis with supplementary materials
    print("\nRunning complete analysis with supplementary materials generation...")
    results_and_supp = analyze_complete_knowledge_graph("database-all-triples.csv")
    
    if results_and_supp and len(results_and_supp) == 2:
        results, supplementary_dir = results_and_supp
        print(f"\n✅ Analysis completed successfully!")
        print(f"✅ Statistical results available in results object")
        print(f"✅ Supplementary materials saved to: {supplementary_dir}")
        print(f"\n📁 Supplementary files created:")
        print(f"   • Table S1: Statistical Results Summary")
        print(f"   • Table S2-S4: Node Identification Comparison")
        print(f"   • Table S5: Node Categories Analysis")
        print(f"   • Table S6: Network Statistics")
        print(f"   • Figure S1: Network Visualization")
        print(f"   • Figure S2: Enrichment Analysis")
        print(f"   • Figure S3: Node Categories")
        print(f"   • Figure S4: Connectivity Heatmap")
        print(f"   • Figure S5: P-value Distribution")
        print(f"\n🎯 Ready for manuscript submission!")
    else:
        print("\n❌ Analysis failed. Check the data and parameters.")

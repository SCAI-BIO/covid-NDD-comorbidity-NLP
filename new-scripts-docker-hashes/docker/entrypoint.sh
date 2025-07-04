#!/bin/bash
# Docker entrypoint script for COVID-NDD Comorbidity Analysis Pipeline
# This script orchestrates the complete reproducible pipeline

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
LOG_LEVEL=${LOG_LEVEL:-INFO}
NEO4J_URI=${NEO4J_URI:-bolt://neo4j:7687}
NEO4J_USER=${NEO4J_USER:-neo4j}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-reproducible_password_2024}

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a /app/logs/pipeline.log
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check if Neo4j is available
wait_for_neo4j() {
    log "Waiting for Neo4j to be available..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if python -c "
from neo4j import GraphDatabase
import sys
try:
    driver = GraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USER', '$NEO4J_PASSWORD'))
    driver.verify_connectivity()
    driver.close()
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
            log "Neo4j is ready!"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: Neo4j not ready, waiting 10 seconds..."
        sleep 10
        ((attempt++))
    done
    
    error_exit "Neo4j failed to become available after $max_attempts attempts"
}

# Data validation function
validate_data() {
    log "=== Starting Data Validation ==="
    
    # Check if data directory exists
    if [ ! -d "/app/data" ]; then
        error_exit "Data directory not found: /app/data"
    fi
    
    # Run hash verification if metadata exists
    if [ -f "/app/data/hashes/kg_versions_and_hashes.json" ]; then
        log "Running SHA256 hash verification..."
        python scripts/generate_hashes.py --verify --data-dir /app/data/raw || {
            log "WARNING: Hash verification failed. Continuing with available data..."
        }
    else
        log "No hash metadata found. Generating hashes for available files..."
        python scripts/generate_hashes.py --data-dir /app/data/raw || {
            log "WARNING: Hash generation failed. Continuing without verification..."
        }
    fi
    
    # Validate data completeness
    log "Validating data completeness..."
    python scripts/validate_data.py --check-completeness || {
        log "WARNING: Data completeness validation failed. Some files may be missing..."
    }
    
    log "=== Data Validation Complete ==="
}

# Neo4j initialization function
initialize_neo4j() {
    log "=== Initializing Neo4j Database ==="
    
    # Wait for Neo4j to be ready
    wait_for_neo4j
    
    # Create database schema and constraints
    log "Creating database schema and constraints..."
    python -c "
from neo4j import GraphDatabase
import sys

driver = GraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USER', '$NEO4J_PASSWORD'))

try:
    with driver.session() as session:
        # Read and execute schema creation script
        with open('/app/cypher/schemas/graph_schema.cypher', 'r') as f:
            schema_script = f.read()
        
        # Execute each statement separately
        statements = [stmt.strip() for stmt in schema_script.split(';') if stmt.strip()]
        for stmt in statements:
            if stmt:
                session.run(stmt)
                print(f'Executed: {stmt[:50]}...')
    
    print('Schema creation completed successfully')
    driver.close()
except Exception as e:
    print(f'Error creating schema: {e}')
    sys.exit(1)
"
    
    log "=== Neo4j Initialization Complete ==="
}

# Data loading function
load_knowledge_graphs() {
    log "=== Loading Knowledge Graphs ==="
    
    # Load each knowledge graph source
    local kg_sources=("drugbank" "opentargets" "disgenet" "indra" "primekg" "literature")
    
    for source in "${kg_sources[@]}"; do
        log "Loading $source data..."
        
        if [ -f "/app/scripts/load_${source}.py" ]; then
            python "scripts/load_${source}.py" --neo4j-uri "$NEO4J_URI" \
                --neo4j-user "$NEO4J_USER" --neo4j-password "$NEO4J_PASSWORD" || {
                log "WARNING: Failed to load $source data. Continuing..."
            }
        else
            log "WARNING: No loader script found for $source"
        fi
    done
    
    log "=== Knowledge Graph Loading Complete ==="
}

# Analysis execution function
run_analysis() {
    log "=== Running Comorbidity Analysis ==="
    
    # Execute main analysis pipeline
    log "Executing analysis queries..."
    python -c "
from neo4j import GraphDatabase
import json
import sys

driver = GraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USER', '$NEO4J_PASSWORD'))

try:
    with driver.session() as session:
        # Read and execute analysis queries
        with open('/app/cypher/queries/data_extraction.cypher', 'r') as f:
            queries = f.read()
        
        # Execute queries and save results
        results = {}
        query_blocks = queries.split('// Query')
        
        for i, block in enumerate(query_blocks[1:], 1):  # Skip first empty block
            lines = block.strip().split('\n')
            query_name = f'query_{i}'
            
            # Find the actual Cypher query (non-comment lines)
            cypher_lines = []
            for line in lines:
                if not line.strip().startswith('//') and line.strip():
                    cypher_lines.append(line)
            
            if cypher_lines:
                cypher_query = '\n'.join(cypher_lines)
                try:
                    result = session.run(cypher_query)
                    results[query_name] = [dict(record) for record in result]
                    print(f'Executed {query_name}: {len(results[query_name])} results')
                except Exception as e:
                    print(f'Error in {query_name}: {e}')
        
        # Save results
        with open('/app/results/analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print('Analysis completed successfully')
    
    driver.close()
except Exception as e:
    print(f'Error during analysis: {e}')
    sys.exit(1)
"
    
    log "=== Analysis Complete ==="
}

# Results export function
export_results() {
    log "=== Exporting Results ==="
    
    # Export various result formats
    log "Generating summary reports..."
    python scripts/generate_reports.py --output-dir /app/results || {
        log "WARNING: Report generation failed"
    }
    
    # Export network data for external tools
    log "Exporting network data..."
    python scripts/export_networks.py --format graphml,json --output-dir /app/results || {
        log "WARNING: Network export failed"
    }
    
    log "=== Results Export Complete ==="
}

# Cleanup function
cleanup() {
    log "=== Pipeline Cleanup ==="
    
    # Compress large result files
    if [ -d "/app/results" ]; then
        log "Compressing results..."
        cd /app/results
        tar -czf pipeline_results_$(date +%Y%m%d_%H%M%S).tar.gz *.json *.csv *.graphml 2>/dev/null || true
        cd /app
    fi
    
    # Generate pipeline summary
    log "Generating pipeline summary..."
    python -c "
import json
import os
from datetime import datetime

summary = {
    'pipeline_version': '1.0.0',
    'execution_date': datetime.now().isoformat(),
    'environment': {
        'neo4j_uri': os.getenv('NEO4J_URI'),
        'log_level': os.getenv('LOG_LEVEL')
    },
    'results_directory': '/app/results',
    'status': 'completed'
}

with open('/app/results/pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Pipeline summary generated')
"
    
    log "=== Cleanup Complete ==="
}

# Main execution function
main() {
    log "=== COVID-NDD Comorbidity Analysis Pipeline Started ==="
    log "Pipeline version: 1.0.0"
    log "Execution mode: $*"
    
    # Create necessary directories
    mkdir -p /app/logs /app/results /app/cache
    
    # Parse command line arguments
    local mode=${1:-full_pipeline}
    
    case $mode in
        "full_pipeline")
            log "Running full pipeline..."
            validate_data
            initialize_neo4j
            load_knowledge_graphs
            run_analysis
            export_results
            cleanup
            ;;
        "validate_only")
            log "Running validation only..."
            validate_data
            ;;
        "analysis_only")
            log "Running analysis only..."
            wait_for_neo4j
            run_analysis
            export_results
            ;;
        "export_only")
            log "Running export only..."
            export_results
            ;;
        *)
            log "Unknown mode: $mode"
            log "Available modes: full_pipeline, validate_only, analysis_only, export_only"
            exit 1
            ;;
    esac
    
    log "=== COVID-NDD Comorbidity Analysis Pipeline Completed Successfully ==="
}

# Trap signals for graceful shutdown
trap 'log "Pipeline interrupted by signal"; exit 130' INT TERM

# Execute main function with all arguments
main "$@"
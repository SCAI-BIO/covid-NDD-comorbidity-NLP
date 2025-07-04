#Draw plots: paths showing comorbidity (showing node degree as well)

import matplotlib.pylab as plt
import networkx as nx
import matplotlib.cm as cm
import numpy as np

# Set up figure size and layout for longer node names
plt.rcParams["figure.figsize"] = [16, 16]
plt.rcParams["figure.autolayout"] = True

# Create an undirected graph
G = nx.Graph()

# Add original nodes and edges
new_edges = [
    ("COVID-19", "Inflammation"),
    ("Inflammation", "MOG"),
    ("MOG", "Encephalomyelitis, Acute Disseminated"),
    ("COVID-19", "Inflammation"),
    ("Inflammation", "IgG"),
    ("IgG", "Immune System Disease"),
    ("COVID-19", "Inflammation"),
    ("Inflammation", "IgG"),
    ("IgG", "Immune System Disease"),
    ("Immune System Disease", "Myelitis"),
    ("Myelitis", "Lymphopenia"),
    ("Lymphopenia", "MOG"),
    ("MOG", "Optic Neuritis"),
    ("COVID-19", "Inflammation"),
    ("Inflammation", "IFNG"),
    ("IFNG", "IL1B"),
    ("IL1B", "Taupaties"),
    ("SARS-CoV-2", "Myelitis"),
    ("Myelitis", "Peripheral Nervous System Diseases"),
    ("SARS-CoV-2", "IL6"),
    ("IL6", "Depression"),
    ("Depression", "TNF"),
    ("TNF", "Neurotransmitters"),
    ("Neurotransmitters", "Encephalitis")
]

# Add edges to the graph
G.add_edges_from(new_edges)

# Generate layout positions
pos = nx.spring_layout(G, k=1.2, seed=42)

# Compute node degrees and normalize for node size and color scaling
degrees = dict(G.degree())
node_size = [v * 300 for v in degrees.values()]  # scale node size by degree
node_color = [v for v in degrees.values()]  # node color based on degree
norm = plt.Normalize(vmin=min(node_color), vmax=max(node_color))  # normalize colors

# Draw nodes with gradient coloring and varied sizes
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, cmap=cm.viridis, alpha=0.9)

# Draw curved edges with transparency to reduce clutter
nx.draw_networkx_edges(G, pos, connectionstyle="arc3,rad=0.4", alpha=0.5)

# Draw labels with larger font for clarity
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_color="black", font_weight="bold")

# Add the title with styling
plt.title("Sherpa", fontsize=16, fontweight='bold', color='darkblue')

# Add a colorbar to show the degree of the nodes
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])
plt.colorbar(sm, label="Node Degree")

plt.show()
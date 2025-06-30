##visualize above using an excel file of the graph statistics 
import pandas as pd
import matplotlib.pyplot as plt

# Updated data based on the new input
data_updated = {
    "source": ["CBM", "DisGeNet", "DrugBank", "INDRA", "OpenTargets", "PubTator3", "SCAI-DMaps", "Sherpa", "PrimeKG"],
    "total_nodes": [4020, 1176, 128, 2004, 31644, 164, 13676, 1516, 258750],
    "unique_nodes": [2010, 588, 64, 1002, 15822, 82, 6838, 758, 129375],
    "unique_triples": [3484, 1048, 710, 1224, 29953, 75, 12302, 1757, 4960573],
    "total_triples": [7843, 2590, 1456, 3049, 72420, 156, 30597, 5018, 16200996],
    "density": [0.000431285, 0.001516862, 0.087352362, 0.000609864, 5.98276E-05, 0.005611252, 0.000131559, 0.001529995, 0.000148185]
}

# Create DataFrame with corrected names
df_updated = pd.DataFrame(data_updated)

# Sort the DataFrame by density in decreasing order
df_sorted_updated = df_updated.sort_values(by="density", ascending=False)

# Set the figure size
plt.figure(figsize=(18, 12))

# Bar plot for total_nodes, unique_nodes, unique_triples, and total_triples sorted by density with logarithmic scale
plt.subplot(2, 2, 1)
df_sorted_updated.plot(x="source", y=["total_nodes", "unique_nodes", "unique_triples", "total_triples"], 
                       kind="bar", ax=plt.gca(), log=True)
plt.title("Total Nodes, Unique Nodes, Unique Triples, Total Triples (Sorted by Density, Log Scale)")
plt.ylabel("Log Count")
plt.xticks(rotation=45)

# Scatter plot for density vs total_nodes (sorted by density)
plt.subplot(2, 2, 2)
plt.scatter(df_sorted_updated["total_nodes"], df_sorted_updated["density"], label="Density vs Total Nodes", color="b")
plt.xlabel("Total Nodes")
plt.ylabel("Density")
plt.title("Density vs Total Nodes (Sorted by Density)")
for i, txt in enumerate(df_sorted_updated["source"]):
    plt.annotate(txt, (df_sorted_updated["total_nodes"].iloc[i], df_sorted_updated["density"].iloc[i]))

# Stacked bar chart for unique_triples and redundant triples sorted by density with logarithmic scale
plt.subplot(2, 2, 3)
df_sorted_updated["redundant_triples"] = df_sorted_updated["total_triples"] - df_sorted_updated["unique_triples"]
df_sorted_updated.plot(x="source", y=["unique_triples", "redundant_triples"], kind="bar", stacked=True, ax=plt.gca(), log=True)
plt.title("Unique vs Redundant Triples (Sorted by Density, Log Scale)")
plt.ylabel("Log Count")
plt.xticks(rotation=45)

# Heatmap for density sorted by density
plt.subplot(2, 2, 4)
plt.bar(df_sorted_updated["source"], df_sorted_updated["density"], color="hotpink")
plt.title("Density Heatmap by Source (Sorted by Density)")
plt.xticks(rotation=45)
plt.ylabel("Density")

# Adjust layout for readability
plt.tight_layout()

# Show the plots
plt.show()
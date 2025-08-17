import json
import os

# Define file paths
KNOWLEDGE_BASE_FILE = "knowledgebase_processed.json"
FIGURES_FILE = "chapter_figure_processed.json"
OUTPUT_FILE = "merged_chunks_with_figures.json"

# Load the two JSON files
with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

with open(FIGURES_FILE, "r", encoding="utf-8") as f:
    figures_data = json.load(f)

print("📘 Loaded knowledge base and figures data.")

# Create a case-insensitive lookup for figures data
figures_lookup = {key.lower(): value for key, value in figures_data.items()}

merged_chunks = []
processed_count = 0

# Iterate through the knowledge base and merge figures
for chunk in knowledge_base:
    subchapter_name = chunk.get("subchapter", "").strip()
    
    # Check for a matching subchapter, case-insensitively
    matching_figures = figures_lookup.get(subchapter_name.lower(), [])
    
    # Add the 'figures' list to the chunk, creating a new field
    chunk["figures"] = matching_figures
    merged_chunks.append(chunk)
    processed_count += 1

print(f"🧩 Merged figures into {processed_count} chunks.")

# Save the merged data to a new file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged_chunks, f, indent=4, ensure_ascii=False)

print(f"✅ Merged file saved to {OUTPUT_FILE}")

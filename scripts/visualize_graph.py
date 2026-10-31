"""Graph Visualization Generator for ExamRAG Knowledge Graph.

Renders high-resolution publication-quality PNG diagrams of academic concept clusters
with color-coded entity types and labeled semantic relations.
"""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from app.graph.store import KnowledgeGraphStore

REPORTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "reports"
GRAPH_JSON_PATH = REPORTS_DIR / "knowledge_graph.json"
OUTPUT_PNG_PATH = REPORTS_DIR / "graph_visualization.png"

# Color palette mapped by EntityType
TYPE_COLORS = {
    "Concept": "#4299E1",      # Blue
    "Topic": "#ED8936",        # Orange
    "Formula": "#48BB78",      # Green
    "Definition": "#9F7AEA",   # Purple
    "Chapter": "#E53E3E"       # Crimson Red
}


def build_sample_academic_subgraph() -> nx.DiGraph:
    """Build a representative curricular subgraph if full graph JSON is pending."""
    g = nx.DiGraph()

    entities = [
        # Chapter 1: Scheduling
        ("cpu_scheduling", "CPU Scheduling", "Topic"),
        ("round_robin", "Round Robin", "Concept"),
        ("time_quantum", "Time Quantum", "Concept"),
        ("fcfs", "First-Come First-Served", "Concept"),
        ("convoy_effect", "Convoy Effect", "Definition"),
        ("sjf", "Shortest Job First", "Concept"),
        ("starvation", "Starvation", "Definition"),

        # Chapter 2: Memory
        ("memory_management", "Memory Management", "Chapter"),
        ("paging", "Paging", "Topic"),
        ("page_table", "Page Table", "Concept"),
        ("tlb", "TLB", "Concept"),
        ("mmu", "MMU", "Concept"),
        ("emat", "EMAT", "Formula"),
        ("internal_frag", "Internal Fragmentation", "Definition"),
        ("external_frag", "External Fragmentation", "Definition"),

        # Chapter 3: Deadlocks
        ("deadlock", "Deadlock", "Topic"),
        ("coffman_conditions", "Coffman Conditions", "Definition"),
        ("circular_wait", "Circular Wait", "Definition"),
        ("bankers_algorithm", "Banker's Algorithm", "Concept"),
        ("dining_philosophers", "Dining Philosophers", "Concept"),
        ("semaphore", "Semaphore", "Concept")
    ]

    for nid, name, etype in entities:
        g.add_node(nid, name=name, entity_type=etype)

    relations = [
        # Scheduling
        ("round_robin", "cpu_scheduling", "PART_OF"),
        ("fcfs", "cpu_scheduling", "PART_OF"),
        ("sjf", "cpu_scheduling", "PART_OF"),
        ("time_quantum", "round_robin", "PREREQUISITE_OF"),
        ("convoy_effect", "fcfs", "DEFINED_IN"),
        ("round_robin", "convoy_effect", "RELATED_TO"),
        ("sjf", "starvation", "RELATED_TO"),

        # Memory
        ("paging", "memory_management", "PART_OF"),
        ("page_table", "paging", "PART_OF"),
        ("tlb", "mmu", "PART_OF"),
        ("tlb", "emat", "PREREQUISITE_OF"),
        ("page_table", "emat", "PREREQUISITE_OF"),
        ("paging", "internal_frag", "RELATED_TO"),
        ("paging", "external_frag", "RELATED_TO"),

        # Deadlocks
        ("coffman_conditions", "deadlock", "DEFINED_IN"),
        ("circular_wait", "coffman_conditions", "PART_OF"),
        ("bankers_algorithm", "deadlock", "RELATED_TO"),
        ("dining_philosophers", "deadlock", "RELATED_TO"),
        ("semaphore", "dining_philosophers", "PREREQUISITE_OF"),
        ("semaphore", "deadlock", "RELATED_TO")
    ]

    for src, tgt, rel in relations:
        g.add_edge(src, tgt, relation=rel)

    return g


def export_graph_visualization(
    graph_store: Optional[KnowledgeGraphStore] = None,
    output_path: Path = OUTPUT_PNG_PATH
) -> Path:
    """Render and save knowledge graph topology diagram."""
    if graph_store is not None and graph_store.graph.number_of_nodes() > 5:
        g = graph_store.graph
    elif GRAPH_JSON_PATH.exists():
        store = KnowledgeGraphStore()
        store.load_json(GRAPH_JSON_PATH)
        g = store.graph
    else:
        g = build_sample_academic_subgraph()

    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

    # Spring layout with tuned parameters for visual separation
    pos = nx.spring_layout(g, k=0.9, iterations=80, seed=42)

    # Assign node colors
    node_colors = []
    labels = {}
    for node, data in g.nodes(data=True):
        etype = data.get("entity_type", "Concept")
        node_colors.append(TYPE_COLORS.get(etype, "#4299E1"))
        labels[node] = data.get("name", node)

    # Draw nodes
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=2400,
        edgecolors="#2D3748",
        linewidths=1.8,
        alpha=0.95
    )

    # Draw node labels
    nx.draw_networkx_labels(
        g,
        pos,
        labels=labels,
        ax=ax,
        font_size=8,
        font_weight="bold",
        font_family="sans-serif",
        font_color="#1A202C"
    )

    # Draw directed edges
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        arrowstyle="-|>",
        arrowsize=14,
        edge_color="#718096",
        width=1.4,
        alpha=0.75,
        connectionstyle="arc3,rad=0.08"
    )

    # Edge labels (relation types)
    edge_labels = {(u, v): d.get("relation", "") for u, v, d in g.edges(data=True)}
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=6,
        font_color="#4A5568",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=color, label=etype)
        for etype, color in TYPE_COLORS.items()
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        facecolor="#F7FAFC",
        edgecolor="#CBD5E0",
        fontsize=9,
        title="Entity Types",
        title_fontproperties={"weight": "bold", "size": 10}
    )

    ax.set_title(
        "ExamRAG Academic Knowledge Graph: Relational Multi-Hop Ontology",
        fontsize=14,
        fontweight="bold",
        color="#1A202C",
        pad=16
    )
    ax.axis("off")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Exported knowledge graph visualization: {output_path}")
    return output_path


if __name__ == "__main__":
    export_graph_visualization()

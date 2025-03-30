"""
Spotify Song Similarity Visualization Program (Optimized)
"""

from __future__ import annotations
import csv
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple

class _Vertex:
    """A vertex representing a song with audio feature similarity calculations."""

    def __init__(self, data: dict, neighbours: set[_Vertex]) -> None:
        self.data = data
        self.neighbours = neighbours

    def audio_feature_similarity(self, other: _Vertex) -> float:
        """Compute cosine similarity between two songs based on audio features."""
        features = ['danceability', 'energy', 'valence',
                   'acousticness', 'speechiness', 'tempo']
        vec1 = [self.data[feature] for feature in features]
        vec2 = [other.data[feature] for feature in features]

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a ** 2 for a in vec1) ** 0.5
        norm2 = sum(b ** 2 for b in vec2) ** 0.5
        return dot / (norm1 * norm2 + 1e-10)

def row_to_track_data(row: dict) -> dict:
    """Convert CSV row to properly typed song data dictionary."""
    return {
        'track_id': str(row['track_id']),
        'track_name': str(row['track_name']),
        'track_artist': str(row['track_artist']),
        'danceability': float(row['danceability']),
        'energy': float(row['energy']),
        'key': int(row['key']),
        'loudness': float(row['loudness']),
        'mode': int(row['mode']),
        'speechiness': float(row['speechiness']),
        'acousticness': float(row['acousticness']),
        'instrumentalness': float(row['instrumentalness']),
        'liveness': float(row['liveness']),
        'valence': float(row['valence']),
        'tempo': float(row['tempo']),
        'duration_ms': int(row['duration_ms'])
    }

class Graph:
    """Graph representing Spotify songs and their similarities."""

    def __init__(self) -> None:
        self._vertices: dict[str, _Vertex] = {}

    @property
    def vertices(self) -> dict[str, _Vertex]:
        return self._vertices

    def add_vertex(self, song_data: dict) -> None:
        song_id = str(song_data['track_id'])
        if song_id not in self._vertices:
            self._vertices[song_id] = _Vertex(song_data, set())

    def add_edge(self, track1: str, track2: str) -> None:
        if track1 not in self._vertices or track2 not in self._vertices:
            raise ValueError("Tracks not found")
        v1 = self._vertices[track1]
        v2 = self._vertices[track2]
        v1.neighbours.add(v2)
        v2.neighbours.add(v1)

    def get_similarity_scores(self, input_song_id: str) -> List[Tuple[str, float]]:
        """Get similarity scores for all songs relative to input."""
        if input_song_id not in self._vertices:
            raise ValueError("Input song not found")

        input_vertex = self._vertices[input_song_id]
        scores = []

        for song_id, other_vtx in self.vertices.items():
            if song_id != input_song_id:
                similarity = input_vertex.audio_feature_similarity(other_vtx)
                scores.append((song_id, similarity))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def get_top_neighbours(self, input_song_id: str, top_n: int = 20) -> set[str]:
        """Get top N most similar songs."""
        scores = self.get_similarity_scores(input_song_id)
        return {song_id for song_id, _ in scores[:top_n]}

def load_song_graph() -> Graph:
    """Load all songs from CSV into graph."""
    print("🔄 Loading songs...")
    graph = Graph()
    with open("spotify_songs.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            song_data = row_to_track_data(row)
            graph.add_vertex(song_data)
    print("✅ Loaded successfully!")
    return graph

def visualize_focused_graph(graph: Graph, input_song_id: str, top_n: int = 20) -> None:
    """Visualize input song and top N similar connections."""
    print("\n🎨 Generating focused visualization...")
    plt.switch_backend('TkAgg')

    # Get top N similar songs
    top_songs = graph.get_top_neighbours(input_song_id, top_n)
    relevant_nodes = {input_song_id}.union(top_songs)

    # Create subgraph
    subgraph = nx.Graph()
    for song_id in relevant_nodes:
        vertex = graph.vertices[song_id]
        subgraph.add_node(song_id, **vertex.data)
        for neighbor in vertex.neighbours:
            if neighbor.data['track_id'] in relevant_nodes:
                subgraph.add_edge(song_id, neighbor.data['track_id'])

    # Create labels and colors
    labels = {
        node: f"{subgraph.nodes[node]['track_name'][:15]}...\n"
              f"({subgraph.nodes[node]['track_artist'][:15]}...)"
        for node in subgraph.nodes
    }
    node_colors = ['red' if node == input_song_id else 'green'
                  for node in subgraph.nodes]

    # Calculate layout
    print("📐 Calculating optimized layout...")
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)

    # Draw graph
    plt.figure(figsize=(16, 12))
    nx.draw(subgraph, pos, labels=labels, node_color=node_colors,
           node_size=1500, font_size=9, edge_color='gray', width=0.8,
           font_weight='bold', alpha=0.9)

    plt.title(f"Top {top_n} Similar Songs to\n"
             f"{graph.vertices[input_song_id].data['track_name']}",
             fontsize=14, pad=20)
    plt.box(False)

    print("🚀 Rendering visualization...")
    plt.get_current_fig_manager().window.wm_geometry("+50+50")
    plt.show(block=True)
    print("✅ Visualization ready! Close window to exit.")

if __name__ == '__main__':
    try:
        # Load data
        song_graph = load_song_graph()

        # User input
        user_input = "6f807x0ima9a1j3VPbc7VN"
        print(f"\n🎵 Processing: {user_input}")

        # Visualize top 25 similar songs
        visualize_focused_graph(song_graph, user_input, top_n=25)

        # Print similarity scores
        print("\n🔢 Top similarity scores:")
        scores = song_graph.get_similarity_scores(user_input)[:25]
        for idx, (song_id, score) in enumerate(scores, 1):
            track = song_graph.vertices[song_id].data
            print(f"{idx:2d}. {score:.2f} | {track['track_name'][:30]}... by {track['track_artist'][:20]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    # Optional code analysis
    import python_ta
    python_ta.check_all(config={
        'max-line-length': 120,
        'disable': ['E1136', 'R0902', 'R0913'],
        'extra-imports': ['csv', 'networkx', 'matplotlib.pyplot'],
        'allowed-io': ['load_song_graph']
    })

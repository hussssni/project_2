from __future__ import annotations
import csv
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple


class _Vertex:
    """Represents a song node in the similarity graph."""

    def __init__(self, data: dict, neighbours: set[_Vertex]) -> None:
        self.data = data
        self.neighbours = neighbours

    def audio_feature_similarity(self, other: _Vertex) -> float:
        """Calculate cosine similarity between audio features."""
        features = ['danceability', 'energy', 'valence',
                    'acousticness', 'speechiness', 'tempo']
        vec1 = [self.data[f] for f in features]
        vec2 = [other.data[f] for f in features]
        dot = sum(a*b for a,b in zip(vec1, vec2))
        norm = (sum(a**2 for a in vec1)**0.5) * (sum(b**2 for b in vec2)**0.5)
        return dot / (norm + 1e-10)

def row_to_track_data(row: dict) -> dict:
    """Convert CSV row to standardized song dictionary."""
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
    """Graph structure for song similarity analysis."""

    def __init__(self) -> None:
        self._vertices: dict[str, _Vertex] = {}

    @property
    def vertices(self) -> dict[str, _Vertex]:
        """Get all vertices in the graph."""
        return self._vertices

    def add_vertex(self, song_data: dict) -> None:
        """Add a song vertex to the graph."""
        song_id = song_data['track_id']
        if song_id not in self._vertices:
            self._vertices[song_id] = _Vertex(song_data, set())

    def get_combined_similarity(self, song1_id: str, song2_id: str, top_n: int = 25) -> List[Tuple[str, float]]:
        """Get songs similar to both inputs using average similarity scores."""
        scores1 = self.get_similarity_scores(song1_id)
        scores2 = self.get_similarity_scores(song2_id)

        score_map = {}
        for tid, score in scores1 + scores2:
            if tid in [song1_id, song2_id]:
                continue
            score_map[tid] = (score_map.get(tid, 0) + score) / 2

        return sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def get_similarity_scores(self, song_id: str) -> List[Tuple[str, float]]:
        """Get similarity scores relative to a track."""
        if song_id not in self._vertices:
            raise ValueError("Track not found")
        vertex = self._vertices[song_id]
        return [(tid, vertex.audio_feature_similarity(v))
                for tid, v in self.vertices.items() if tid != song_id]

    def find_song_id_by_name(self, name: str) -> str:
        """Find track ID by partial name match."""
        matches = []
        lower_name = name.lower()
        for tid, v in self._vertices.items():
            if lower_name in v.data['track_name'].lower():
                matches.append((tid, v.data))

        if not matches:
            raise ValueError(f"No matches found for: {name}")
        if len(matches) > 1:
            print(f"Multiple matches ({len(matches)}):")
            for i, (_, data) in enumerate(matches, 1):
                print(f"{i}. {data['track_name']} by {data['track_artist']}")
            while True:
                try:
                    choice = int(input("Enter selection: "))
                    if 1 <= choice <= len(matches):
                        return matches[choice-1][0]
                except ValueError:
                    print("Invalid input")
        return matches[0][0]

def load_song_graph() -> Graph:
    """Initialize graph from CSV data."""
    graph = Graph()
    with open("spotify_songs.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            graph.add_vertex(row_to_track_data(row))
    return graph

def visualize_dual_similarity(graph: Graph, song1_id: str, song2_id: str, top_n: int = 25) -> None:
    """Visualize songs similar to both input tracks."""
    combined = graph.get_combined_similarity(song1_id, song2_id, top_n)
    nodes = {song1_id, song2_id}.union({tid for tid, _ in combined})

    subgraph = nx.Graph()
    for tid in nodes:
        vertex = graph.vertices[tid]
        subgraph.add_node(tid, **vertex.data)
        for neighbor in vertex.neighbours:
            if neighbor.data['track_id'] in nodes:
                subgraph.add_edge(tid, neighbor.data['track_id'])

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(subgraph, k=0.6, seed=42)
    colors = ['#FF6B6B' if n == song1_id else
             '#4ECDC4' if n == song2_id else
             '#A0DAA9' for n in subgraph.nodes]

    nx.draw(subgraph, pos, node_color=colors,
           labels={n: f"{subgraph.nodes[n]['track_name'][:15]}..." for n in subgraph.nodes},
           node_size=1200, font_size=8, edge_color='#556270',
           width=1.2, alpha=0.9)

    plt.title(f"Songs Similar to Both\n"
             f"{graph.vertices[song1_id].data['track_name']}\nand\n"
             f"{graph.vertices[song2_id].data['track_name']}")
    plt.show()

if __name__ == '__main__':
    try:
        graph = load_song_graph()

        song1 = input("Enter first song name: ")
        song1_id = graph.find_song_id_by_name(song1)
        song2 = input("Enter second song name: ")
        song2_id = graph.find_song_id_by_name(song2)

        print(f"\nBase tracks:")
        print(f"1. {graph.vertices[song1_id].data['track_name']}")
        print(f"2. {graph.vertices[song2_id].data['track_name']}")

        combined = graph.get_combined_similarity(song1_id, song2_id)
        visualize_dual_similarity(graph, song1_id, song2_id)

        print("\nTop similar tracks:")
        for i, (tid, score) in enumerate(combined, 1):
            track = graph.vertices[tid].data
            print(f"{i:2d}. {score:.2f} | {track['track_name']} by {track['track_artist']}")

    except Exception as e:
        print(f"Error: {e}")

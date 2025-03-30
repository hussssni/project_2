from __future__ import annotations
from typing import List, Tuple
import csv
import networkx as nx
import matplotlib.pyplot as plt


class _Vertex:
    """Represents a song vertex in the similarity graph.

    Instance Attributes:
        data: Dictionary containing song metadata and audio features
        neighbours: Set of connected vertex objects
    """

    def __init__(self, data: dict, neighbours: set[_Vertex]) -> None:
        self.data = data
        self.neighbours = neighbours

    def audio_feature_similarity(self, other: _Vertex) -> float:
        """Calculate cosine similarity between two songs' audio features.
        """
        features = ['danceability', 'energy', 'valence',
                    'acousticness', 'speechiness', 'tempo']
        vec1 = [self.data[feature] for feature in features]
        vec2 = [other.data[feature] for feature in features]
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a ** 2 for a in vec1) ** 0.5
        norm2 = sum(b ** 2 for b in vec2) ** 0.5
        return dot / (norm1 * norm2 + 1e-10)


def row_to_track_data(row: dict) -> dict:
    """Convert CSV row data to standardized song dictionary.
    """
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
    """Main graph structure for storing and analyzing song relationships.

    Instance Attributes:
        _vertices: Dictionary mapping track IDs to Vertex objects
    """

    def __init__(self) -> None:
        self._vertices: dict[str, _Vertex] = {}

    @property
    def vertices(self) -> dict[str, _Vertex]:
        """Get all vertices in the graph.
        """
        return self._vertices

    def add_vertex(self, song_data: dict) -> None:
        """Add a new song vertex to the graph.y
        """
        song_id = str(song_data['track_id'])
        if song_id not in self._vertices:
            self._vertices[song_id] = _Vertex(song_data, set())

    def add_edge(self, track1: str, track2: str) -> None:
        """Create connection between two tracks.
        """
        if track1 not in self._vertices or track2 not in self._vertices:
            raise ValueError("Tracks not found")
        v1 = self._vertices[track1]
        v2 = self._vertices[track2]
        v1.neighbours.add(v2)
        v2.neighbours.add(v1)

    def get_similarity_scores(self, input_song_id: str) -> List[Tuple[str, float]]:
        """Calculate similarity scores for all songs relative to target.
        """
        if input_song_id not in self._vertices:
            raise ValueError("Input song not found")
        input_vertex = self._vertices[input_song_id]
        return sorted(
            [(sid, input_vertex.audio_feature_similarity(v))
             for sid, v in self.vertices.items() if sid != input_song_id],
            key=lambda x: x[1], reverse=True
        )

    def get_top_neighbours(self, input_song_id: str, top_n: int = 20) -> set[str]:
        """Get top N most similar track IDs.
        """
        scores = self.get_similarity_scores(input_song_id)
        return {song_id for song_id, _ in scores[:top_n]}

    def find_song_id_by_name(self, song_name: str) -> str:
        """Locate track ID by song name with partial matching.
        """
        matches = [
            (tid, v.data) for tid, v in self._vertices.items()
            if song_name.lower() in v.data['track_name'].lower()
        ]
        if not matches:
            raise ValueError(f"No songs found matching: {song_name}")
        if len(matches) > 1:
            print(f"Found {len(matches)} matching songs:")
            for i, (_, data) in enumerate(matches, 1):
                print(f"{i}. {data['track_name']} by {data['track_artist']}")
            while True:
                try:
                    choice = int(input("Enter selection number: "))
                    if 1 <= choice <= len(matches):
                        return matches[choice-1][0]
                except ValueError:
                    print("Invalid number")
        return matches[0][0]

def load_song_graph() -> Graph:
    """Initialize graph from Spotify dataset CSV.
    """
    graph = Graph()
    with open("spotify_songs.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            graph.add_vertex(row_to_track_data(row))
    return graph

def visualize_focused_graph(graph: Graph, input_song_id: str, top_n: int = 20) -> None:
    """Generate visualization of similar songs network.
    """
    plt.switch_backend('TkAgg')
    top_songs = graph.get_top_neighbours(input_song_id, top_n)
    relevant_nodes = {input_song_id}.union(top_songs)

    subgraph = nx.Graph()
    for song_id in relevant_nodes:
        vertex = graph.vertices[song_id]
        subgraph.add_node(song_id, **vertex.data)
        for neighbor in vertex.neighbours:
            if neighbor.data['track_id'] in relevant_nodes:
                subgraph.add_edge(song_id, neighbor.data['track_id'])

    pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
    plt.figure(figsize=(16, 12))
    nx.draw(subgraph, pos,
           labels={n: f"{subgraph.nodes[n]['track_name'][:15]}...\n({subgraph.nodes[n]['track_artist'][:15]}...)"
                   for n in subgraph.nodes},
           node_color=['red' if n == input_song_id else 'green' for n in subgraph.nodes],
           node_size=1500, font_size=9, edge_color='gray', width=0.8,
           font_weight='bold', alpha=0.9)
    plt.title(f"Top {top_n} Similar Songs to\n{graph.vertices[input_song_id].data['track_name']}")
    plt.show(block=True)


if __name__ == '__main__':
    try:
        song_graph = load_song_graph()
        search_term = input("Enter song name: ")
        user_id = song_graph.find_song_id_by_name(search_term)
        print(f"Selected: {song_graph.vertices[user_id].data['track_name']}")
        visualize_focused_graph(song_graph, user_id)
        scores = song_graph.get_similarity_scores(user_id)[:25]
        for idx, (sid, score) in enumerate(scores, 1):
            track = song_graph.vertices[sid].data
            print(f"{idx:2d}. {score:.2f} | {track['track_name'][:30]}...")
    except Exception as e:
        print(f"Error: {e}")

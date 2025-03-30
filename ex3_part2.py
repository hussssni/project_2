from __future__ import annotations
from typing import List, Tuple
import csv
import networkx as nx
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import ttk


#########################
# Core Functionality Module
#########################

class _Vertex:
    """Represents a song vertex in the similarity graph.

    Instance Attributes:
        data: Dictionary containing song metadata and audio features.
        neighbours: Set of connected vertex objects.
    """
    def __init__(self, data: dict, neighbours: set[_Vertex]) -> None:
        self.data = data
        self.neighbours = neighbours

    def audio_feature_similarity(self, other: _Vertex) -> float:
        """Calculate cosine similarity between two songs' audio features."""
        features = ['danceability', 'energy', 'valence',
                    'acousticness', 'speechiness', 'tempo']
        vec1 = [self.data[feature] for feature in features]
        vec2 = [other.data[feature] for feature in features]
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a ** 2 for a in vec1) ** 0.5
        norm2 = sum(b ** 2 for b in vec2) ** 0.5
        return dot / (norm1 * norm2 + 1e-10)


def row_to_track_data(row) -> dict:
    """Convert CSV row data to a standardized song dictionary.

    Now also parses the 'track_popularity' field.
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
        'duration_ms': int(row['duration_ms']),
        'track_popularity': int(row['track_popularity'])
    }


class Graph:
    """Main graph structure for storing and analyzing song relationships.

    Instance Attributes:
        _vertices: Dictionary mapping track IDs to _Vertex objects.
    """
    def __init__(self) -> None:
        self._vertices: dict[str, _Vertex] = {}

    @property
    def vertices(self) -> dict[str, _Vertex]:
        """Return all vertices in the graph."""
        return self._vertices

    def add_vertex(self, song_data: dict) -> None:
        """Add a new song vertex to the graph."""
        song_id = str(song_data['track_id'])
        if song_id not in self._vertices:
            self._vertices[song_id] = _Vertex(song_data, set())

    def add_edge(self, track1: str, track2: str) -> None:
        """Create connection between two tracks."""
        if track1 not in self._vertices or track2 not in self._vertices:
            raise ValueError("Tracks not found")
        v1 = self._vertices[track1]
        v2 = self._vertices[track2]
        v1.neighbours.add(v2)
        v2.neighbours.add(v1)

    def get_similarity_score(self, v1: _Vertex, v2: _Vertex) -> float:
        """Calculate similarity score between two songs based on multiple features."""
        features = [
            "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo",
        ]
        feature_ranges = {
            "danceability": (0.0, 1.0),
            "energy": (0.0, 1.0),
            "key": (-1, 11),
            "loudness": (-60, 0),
            "mode": (0, 1),
            "speechiness": (0.0, 1.0),
            "acousticness": (0.0, 1.0),
            "instrumentalness": (0.0, 1.0),
            "liveness": (0.0, 1.0),
            "valence": (0.0, 1.0),
            "tempo": (60, 200),
        }
        diff_squared_sum = 0.0
        for feat in features:
            min_val, max_val = feature_ranges[feat]
            range_val = max_val - min_val
            v1_val = float(v1.data[feat])
            v2_val = float(v2.data[feat])
            scaled_v1 = (v1_val - min_val) / range_val
            scaled_v2 = (v2_val - min_val) / range_val
            diff_squared_sum += (scaled_v1 - scaled_v2) ** 2
        distance = math.sqrt(diff_squared_sum)
        max_distance = math.sqrt(len(features))
        similarity = 1 - (distance / max_distance)
        return max(0, similarity)

    def get_similarity_scores(self, input_song_id: str) -> List[Tuple[str, float]]:
        """Return a sorted list of similarity scores relative to the input song.

        Duplicate songs (same name and artist) are filtered out.
        """
        if input_song_id not in self._vertices:
            raise ValueError("Input song not found")
        input_vertex = self._vertices[input_song_id]
        input_key = (
            input_vertex.data['track_name'].strip().lower(),
            input_vertex.data['track_artist'].strip().lower()
        )
        seen = set()
        scores = []
        for song_id, vertex in self._vertices.items():
            key = (
                vertex.data['track_name'].strip().lower(),
                vertex.data['track_artist'].strip().lower()
            )
            if key == input_key or key in seen:
                continue
            scores.append((song_id, self.get_similarity_score(input_vertex, vertex)))
            seen.add(key)
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def get_top_neighbours(self, input_song_id: str, top_n: int = 20) -> set[str]:
        """Get the top N most similar track IDs (duplicates removed)."""
        if input_song_id not in self._vertices:
            raise ValueError("Input song not found")
        input_vertex = self._vertices[input_song_id]
        input_key = (
            input_vertex.data['track_name'].strip().lower(),
            input_vertex.data['track_artist'].strip().lower()
        )
        scores = {song_id: self.get_similarity_score(input_vertex, vertex)
                  for song_id, vertex in self._vertices.items()}
        scores_sorted = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        filtered = []
        seen = set()
        for song_id, score in scores_sorted.items():
            v = self._vertices[song_id]
            key = (
                v.data['track_name'].strip().lower(),
                v.data['track_artist'].strip().lower()
            )
            if key == input_key or key in seen:
                continue
            seen.add(key)
            filtered.append((song_id, score))
            if len(filtered) >= top_n:
                break
        return {song_id for song_id, _ in filtered}

    def find_song_id_by_name(self, song_name: str) -> str:
        """Find a song ID by name with partial matching, filtering out duplicates.

        Now matches if the query is found in either the track name or the artist name.
        """
        matches = [
            (tid, v.data) for tid, v in self._vertices.items()
            if (song_name.lower() in v.data['track_name'].lower() or
                song_name.lower() in v.data['track_artist'].lower())
        ]
        if not matches:
            raise ValueError(f"No songs found matching: {song_name}")
        unique_matches = {}
        for tid, data in matches:
            key = (data['track_name'].strip().lower(), data['track_artist'].strip().lower())
            if key not in unique_matches:
                unique_matches[key] = (tid, data)
        unique_list = list(unique_matches.values())
        if len(unique_list) > 1:
            print(f"Found {len(unique_list)} unique matching songs:")
            for i, (_, data) in enumerate(unique_list, 1):
                print(f"{i}. {data['track_name']} by {data['track_artist']}")
            while True:
                try:
                    choice = int(input("Enter selection number: "))
                    if 1 <= choice <= len(unique_list):
                        return unique_list[choice - 1][0]
                except ValueError:
                    print("Invalid number")
        return unique_list[0][0]


def load_song_graph() -> Graph:
    """Load the song graph from the CSV dataset,
    filtering out the bottom 10000 least popular songs.
    """
    graph = Graph()
    with open("spotify_songs.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        # Load all rows as songs.
        songs = [row_to_track_data(row) for row in reader]
    # If there are more than 10000 songs, remove the bottom 10000 by popularity.
    if len(songs) > 10000:
        songs.sort(key=lambda song: song['track_popularity'])
        songs = songs[10000:]
    # Add the remaining songs to the graph.
    for song in songs:
        graph.add_vertex(song)
    return graph


def visualize_focused_graph(graph: Graph, input_song_id: str, threshold: float = 0.9) -> None:
    """Visualize the network of similar songs using matplotlib.

    This version builds a subgraph including:
      - The input song.
      - All songs with a similarity score >= threshold (0.9) relative to the input song.
    It then adds edges between:
      - Edges connecting the input song to any other node if similarity >= 0.9.
      - Edges between nodes not including the input song if their similarity >= 0.95.
    """
    plt.switch_backend('TkAgg')
    # Get similarity scores relative to the input song.
    scores = graph.get_similarity_scores(input_song_id)
    # Start with the input song.
    selected_nodes = {input_song_id}
    for song_id, sim in scores:
        if sim >= threshold:
            selected_nodes.add(song_id)

    # Build a subgraph with selected nodes.
    subgraph = nx.Graph()
    for song_id in selected_nodes:
        vertex = graph.vertices[song_id]
        subgraph.add_node(song_id, **vertex.data)

    # Add edges between every pair of nodes using different thresholds.
    selected_list = list(selected_nodes)
    for i in range(len(selected_list)):
        for j in range(i + 1, len(selected_list)):
            node_i = selected_list[i]
            node_j = selected_list[j]
            sim = graph.get_similarity_score(graph.vertices[node_i], graph.vertices[node_j])
            # Use threshold 0.9 if either node is the input song; else 0.95.
            if node_i == input_song_id or node_j == input_song_id:
                edge_threshold = 0.9
            else:
                edge_threshold = 0.95
            if sim >= edge_threshold:
                subgraph.add_edge(node_i, node_j)

    # Use Kamada-Kawai layout for better node distribution.
    pos = nx.kamada_kawai_layout(subgraph)
    plt.figure(figsize=(16, 12))
    labels = {n: f"{subgraph.nodes[n]['track_name'][:15]}...\n({subgraph.nodes[n]['track_artist'][:15]}...)"
              for n in subgraph.nodes}
    colors = ['red' if n == input_song_id else 'green' for n in subgraph.nodes]
    nx.draw(subgraph, pos, labels=labels, node_color=colors, node_size=1500,
            font_size=9, edge_color='gray', width=0.8, font_weight='bold', alpha=0.9)
    plt.title(f"Songs with similarity >= {int(threshold*100)}% (Input edges) and >= 95% (Other edges) to\n{graph.vertices[input_song_id].data['track_name']}")
    plt.show(block=True)


#########################
# GUI Module
#########################

def get_unique_songs(graph: Graph) -> List[Tuple[str, str, str]]:
    """Return a deduplicated list of songs as (track_id, track_name, track_artist) tuples."""
    seen = {}
    for tid, vertex in graph.vertices.items():
        key = (vertex.data['track_name'].strip().lower(), vertex.data['track_artist'].strip().lower())
        if key not in seen:
            seen[key] = (tid, vertex.data['track_name'], vertex.data['track_artist'])
    return list(seen.values())


def run_gui(graph: Graph):
    """Run the Tkinter GUI for song search and recommendations."""
    unique_songs = get_unique_songs(graph)
    root = tk.Tk()
    root.title("Song Finder")

    # Song search entry and suggestions listbox.
    search_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=search_var, width=50)
    entry.pack(pady=10)

    suggestions_listbox = tk.Listbox(root, width=50, height=10)
    suggestions_listbox.pack()

    # Listbox for displaying top 25 recommendations.
    rec_label = tk.Label(root, text="Top 25 Recommendations")
    rec_label.pack(pady=(10, 0))
    recommendations_listbox = tk.Listbox(root, width=80, height=25)
    recommendations_listbox.pack(pady=(0, 10))

    def update_suggestions(*args):
        query = search_var.get().strip().lower()
        suggestions_listbox.delete(0, tk.END)
        for tid, name, artist in unique_songs:
            # Check if the query is in the track name or artist name.
            if query in name.lower() or query in artist.lower():
                suggestions_listbox.insert(tk.END, f"{name} - {artist}")

    search_var.trace_add("write", update_suggestions)

    def on_listbox_select(event):
        selection = suggestions_listbox.curselection()
        if selection:
            index = selection[0]
            value = suggestions_listbox.get(index)
            search_var.set(value)

    suggestions_listbox.bind('<<ListboxSelect>>', on_listbox_select)

    def select_song():
        query = search_var.get().strip()
        for tid, name, artist in unique_songs:
            if f"{name} - {artist}" == query:
                # Visualize the graph with similarity threshold edges.
                visualize_focused_graph(graph, tid, threshold=0.9)
                # Retrieve and display the top 25 recommendations.
                scores = graph.get_similarity_scores(tid)[:25]
                recommendations_listbox.delete(0, tk.END)
                recommendations_listbox.insert(tk.END, f"Selected: {name} - {artist}")
                recommendations_listbox.insert(tk.END, "----------------------------------------")
                for idx, (sid, score) in enumerate(scores, 1):
                    track = graph.vertices[sid].data
                    # Format: Song - Artist | Similarity Rate: 50%
                    recommendations_listbox.insert(
                        tk.END,
                        f"{track['track_name']} - {track['track_artist']} | Similarity Rate: {int(score * 100)}%"
                    )
                return
        recommendations_listbox.delete(0, tk.END)
        recommendations_listbox.insert(tk.END, "Song not found. Please select from the suggestions.")

    btn = tk.Button(root, text="Select Song", command=select_song)
    btn.pack(pady=10)
    root.mainloop()


#########################
# Main Function (main.py)
#########################

def main():
    """Main entry point for the CSC111 Project 2 final submission."""
    try:
        # Load the song graph from the dataset (filtering out bottom 10,000 least popular songs).
        song_graph = load_song_graph()
        # Launch the GUI for interactive song search and recommendations.
        run_gui(song_graph)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

from __future__ import annotations
import csv
from typing import Any

# Make sure you've installed the necessary Python libraries (see assignment handout
# "Installing new libraries" section)
import networkx as nx  # Used for visualizing graphs (by convention, referred to as "nx")


class _Vertex:
    """A vertex in a musical songs graph, used to represent a song.

    Each vertex item is a dictionary.
    ex_vertex = {"track_id" : "asdj", "track_name" : "wasd"...... }

    Instance Attributes:
        - track_id: Unique ID given to each song.
        - track_name: Name of the song.
        - track_artist: Name of artist who sings song.
        - track_popularity: Song popularity from 0-100, where higher is more popular.
        - danceability: How suitable a track is for dancing based on certain musical elements. 0 least -> 1 most.
        - energy: Measure from 0 to 1 representing a measure of intensity and activity. 0 least -> 1 most.
        - key: Estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . 0 = C, 1 = C♯/D♭, 2 = D, and so on.
        - loudness: Overall loudness of a track in decibels. Values range between -60 and 0 db.
        - mode: Indicates modality of a track (major or minor). Major = 1, Minor - 0.
        - speechiness: Detects presence of spoken words in a track. More speaking = closer to 1.
        - acousticness: Confidence measure from 0.0 to 1.0 of whether the tack is acoustic. Higher score -> more acoustic.
        - instrumentalness: Predicts whether a track contains no vocals. Closer to 1 means no vocals.
        - liveness: Detects the presence of a live recording of a song from 0 to 1. Over 0.8 means highly recorded live.
        - valence: Measure from 0 to 1 describing musical positiveness. Closer to 1 means more positive (cheerful).
        - tempo: Estimated tempo in BPM (Beats per minute). Higher tempo means faster song

    Representation Invariants:
        - self not in self.neighbours
        - all(self in u.neighbours for u in self.neighbours)

    """
    def __init__(self, data: Any, neighbours: set[_Vertex]) -> None:
        """Initialize a new vertex with the given variables:
        This vertex is initialized with no neighbours.

        Preconditions:
            - kind in {'user', 'book'} CHANGE
        """

        # self.data = {"track_id": track_id, "track_name": track_name, "track_artist": track_artist,
        #   "track_popularity": track_popularity, "track_danceability": danceability, "energy": energy, }

        self.data = data
        self.neighbours = neighbors

    def degree(self) -> int:
        """Return the degree of this vertex."""
        return len(self.neighbours)

    ############################################################################
    # Part 2, Q2a
    ############################################################################
    def similarity_score(self, other: _Vertex) -> float:
        """Return the similarity score between this vertex and other.

        See Assignment handout for definition of similarity score.
        """
        if len(self.neighbours) == 0 or len(other.neighbours) == 0:
            return 0

        shared = self.neighbours.intersection(other.neighbours)
        union = self.neighbours.union(other.neighbours)

        return len(shared) / len(union)


class Graph:
    """A graph used to represent a book review network.
    """
    # Private Instance Attributes:
    #     - _vertices:
    #         A collection of the vertices contained in this graph.
    #         Maps item to _Vertex object.
    _vertices: dict[Any, _Vertex]

    def __init__(self) -> None:
        """Initialize an empty graph (no vertices or edges)."""
        self._vertices = {}

    def row_to_track_data(row) -> dict:
        """Given a pandas Series `row` representing one song,
        return a dictionary with all relevant fields.
        """
        return {
            'track_id': row['track_id'],
            'track_name': row['track_name'],
            'track_artist': row['track_artist'],
            'danceability': row['danceability'],
            'energy': row['energy'],
            'key': row['key'],
            'loudness': row['loudness'],
            'mode': row['mode'],
            'speechiness': row['speechiness'],
            'acousticness': row['acousticness'],
            'instrumentalness': row['instrumentalness'],
            'liveness': row['liveness'],
            'valence': row['valence'],
            'tempo': row['tempo'],
            'duration_ms': row['duration_ms']
        }

    def add_vertex(self, row: dict, kind: str) -> None:
        """Add a vertex for a single Spotify track from a CSV row.

        The row must contain columns:
          - 'track_id'
          - 'track_name'
          - 'track_artist'
          - 'danceability'
          - 'energy'
          - 'key'
          - 'loudness'
          - 'mode'
          - 'speechiness'
          - 'acousticness'
          - 'instrumentalness'
          - 'liveness'
          - 'valence'
          - 'tempo'
          - 'duration_ms'

        Do nothing if the track_id is already in this graph.

        Preconditions:
            - kind in {'song', ...}  # extend as needed
        """
        # Build a dictionary with the required track information.
        # Since CSV data is read as strings, convert numeric fields to float.

        with open("spotify_songs.csv", "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for line in reader:
                song_data = row_to_track_data(line)
                song_id = song_data['track_id']
                self._vertices[song_id] = song_data.pop('track_id')

    def add_edge(self, track1: Any, track2: Any) -> None:
        """Add an edge between the two vertices with the given items in this graph.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - item1 != item2
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[track1]
            v2 = self._vertices[track2]

            if self.get_similarity_score(v1, v2) >= 0.5:

                v1.neighbours.add(v2)
                v2.neighbours.add(v1)
        else:
            raise ValueError

    def adjacent(self, item1: Any, item2: Any) -> bool:
        """Return whether item1 and item2 are adjacent vertices in this graph.

        Return False if item1 or item2 do not appear as vertices in this graph.
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            return any(v2.item == item2 for v2 in v1.neighbours)
        else:
            return False

    def get_neighbours(self, item: Any) -> set:
        """Return a set of the neighbours of the given item.

        Note that the *items* are returned, not the _Vertex objects themselves.

        Raise a ValueError if item does not appear as a vertex in this graph.
        """
        if item in self._vertices:
            v = self._vertices[item]
            return {neighbour.item for neighbour in v.neighbours}
        else:
            raise ValueError

    def get_all_vertices(self, kind: str = '') -> set:
        """Return a set of all vertex items in this graph.

        If kind != '', only return the items of the given vertex kind.

        Preconditions:
            - kind in {'', 'user', 'book'}
        """
        if kind != '':
            return {v.item for v in self._vertices.values() if v.kind == kind}
        else:
            return set(self._vertices.keys())

    def to_networkx(self, max_vertices: int = 5000) -> nx.Graph:
        """Convert this graph into a networkx Graph.

        max_vertices specifies the maximum number of vertices that can appear in the graph.
        (This is necessary to limit the visualization output for large graphs.)

        Note that this method is provided for you, and you shouldn't change it.
        """
        graph_nx = nx.Graph()
        for v in self._vertices.values():
            graph_nx.add_node(v.item, kind=v.kind)

            for u in v.neighbours:
                if graph_nx.number_of_nodes() < max_vertices:
                    graph_nx.add_node(u.item, kind=u.kind)

                if u.item in graph_nx.nodes:
                    graph_nx.add_edge(v.item, u.item)

            if graph_nx.number_of_nodes() >= max_vertices:
                break

        return graph_nx

    ############################################################################
    # Part 2, Q2b
    ############################################################################
    def get_similarity_score(self, item1: Any, item2: Any) -> float:
        """Return the similarity score between the two given items in this graph.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        >>> g = Graph()
        >>> for i in range(0, 6):
        ...     g.add_vertex(str(i), kind='user')
        >>> g.add_edge('0', '2')
        >>> g.add_edge('0', '3')
        >>> g.add_edge('0', '4')
        >>> g.add_edge('1', '3')
        >>> g.add_edge('1', '4')
        >>> g.add_edge('1', '5')
        >>> g.get_similarity_score('0', '1')
        0.5
        """
        if item1 not in self._vertices or item2 not in self._vertices:
            raise ValueError(f'One or both of the items {item1}, {item2} are not in this graph')
        v1 = self._vertices[item1]
        v2 = self._vertices[item2]

        return v1.similarity_score(v2)

    ############################################################################
    # Part 2, Q3
    ############################################################################
    def recommend_books(self, book: str, limit: int) -> list[str]:
        """Return a list of up to <limit> recommended books based on similarity to the given book.

        The return value is a list of the titles of recommended books, sorted in
        *descending order* of similarity score. Ties are broken in descending order
        of book title. That is, if v1 and v2 have the same similarity score, then
        v1 comes before v2 if and only if v1.item > v2.item.

        The returned list should NOT contain:
            - the input book itself
            - any book with a similarity score of 0 to the input book
            - any duplicates
            - any vertices that represents a user (instead of a book)

        Up to <limit> books are returned, starting with the book with the highest similarity score,
        then the second-highest similarity score, etc. Fewer than <limit> books are returned if
        and only if there aren't enough books that meet the above criteria.

        Preconditions:
            - book in self._vertices
            - self._vertices[book].kind == 'book'
            - limit >= 1
        """
        book_vertex = self._vertices[book]
        possible_matches = []
        for vertex in self._vertices.values():
            if vertex.kind == 'book' and vertex.item != book:
                score = book_vertex.similarity_score(vertex)
                if score > 0:
                    possible_matches.append((score, vertex.item))

        possible_matches.sort(key=lambda x: (x[0], x[1]), reverse=True)

        top_books = possible_matches[:limit]
        return [title for book_score, title in top_books]


################################################################################
# Part 2, Q1
################################################################################
def load_review_graph(reviews_file: str, book_names_file: str) -> Graph:
    """Return a book review graph corresponding to the given datasets.

    The book review graph stores all the information from reviews_file as follows:
    Create one vertex for each user, AND one vertex for each unique book reviewed in the datasets.
    Edges represent a review between a user and a book (that is, you should add an edge between each user and
    all the books that particular user has reviewed).

    The vertices of the 'user' kind should have the user ID as its item.
    The vertices of the 'book' kind representing each reviewed book should have the book TITLE as its item (you should
     use the book_names_file to find the book title associated with each book id).

    Use the "kind" _Vertex attribute to differentiate between the two vertex types.

    Note: In this graph, each edge only represents the existence of a review---IGNORE THE REVIEW SCORE in the
    datasets, as we don't have a way to represent these scores (yet).

    Preconditions:
        - reviews_file is the path to a CSV file corresponding to the book review data
          format described on the assignment handout
        - book_names_file is the path to a CSV file corresponding to the book data
          format described on the assignment handout
        - each book ID in reviews_file exists as a book ID in book_names_file

    >>> g = load_review_graph('data/reviews_small.csv', 'data/book_names.csv')
    >>> len(g.get_all_vertices(kind='book'))
    4
    >>> len(g.get_all_vertices(kind='user'))
    5
    >>> user1_reviews = g.get_neighbours('user1')
    >>> len(user1_reviews)
    3
    >>> "Harry Potter and the Sorcerer's Stone (Book 1)" in user1_reviews
    True
    """
    '''book_id_to_title = {}
    with open('spotify_data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            book_id, book_title = row[0], row[1]
            book_id_to_title[book_id] = book_title

    g = Graph()

    users_added = set()
    books_added = set()
    with open(reviews_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            user_id = row[0]
            book_id = row[1]
            book_title = book_id_to_title[book_id]
            if user_id not in users_added:
                g.add_vertex(user_id, 'user')
                users_added.add(user_id)

            if book_title not in books_added:
                g.add_vertex(book_title, 'book')
                books_added.add(book_title)

            g.add_edge(user_id, book_title)
    '''

    # Assuming `graph` is an instance of your graph class:
    graph = Graph()
    with open("spotify_songs.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            graph.add_vertex(row, kind='song')

    return graph


if __name__ == '__main__':
    # You can uncomment the following lines for code checking/debugging purposes.
    # However, we recommend commenting out these lines when working with the large
    # datasets, as checking representation invariants and preconditions greatly
    # increases the running time of the functions/methods.
    import python_ta.contracts

    python_ta.contracts.check_all_contracts()

    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'max-line-length': 120,
        'disable': ['E1136'],
        'extra-imports': ['csv', 'networkx'],
        'allowed-io': ['load_review_graph'],
        'max-nested-blocks': 4
    })

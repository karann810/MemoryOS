import json
from types import SimpleNamespace

from memory_os.memory import MemoryOS, PointStruct


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if "You merge several memory texts" in prompt:
            return "Merged memory text preserving the distinct facts"
        if "Python" in prompt:
            return json.dumps(
                {
                    "memories": [
                        {
                            "text": "User prefers Python",
                            "importance": 0.95,
                            "emotion": "joy",
                        },
                        {
                            "text": "User is building an auth feature",
                            "importance": 0.8,
                            "emotion": "neutral",
                        },
                    ]
                }
            )
        if "deadline" in prompt:
            return json.dumps(
                {
                    "memories": [
                        {
                            "text": "User is stressed about a launch deadline",
                            "importance": 0.9,
                            "emotion": "fear",
                        }
                    ]
                }
            )
        return json.dumps({"memories": [{"text": "General memory", "importance": 0.7, "emotion": "neutral"}]})


class FakeSentenceTransformer:
    def __init__(self):
        self.texts = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, text, convert_to_numpy=False, normalize_embeddings=True):
        self.texts.append(text)
        return [float(len(text)), 1.0, 0.5]


class FakeQdrantClient:
    def __init__(self, url, api_key=None):
        self.url = url
        self.api_key = api_key
        self.exists = False
        self.points = []
        self.indexed_fields = []

    def collection_exists(self, collection_name):
        return self.exists

    def create_collection(self, collection_name, vectors_config):
        self.exists = True

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.indexed_fields.append((field_name, field_schema))

    def upsert(self, collection_name, points):
        self.points.extend(points)

    def query_points(self, collection_name, query, query_filter=None, limit=10, with_payload=True):
        matches = [SimpleNamespace(id=p.id, payload=p.payload, score=1.0) for p in self.points]
        if query_filter is not None:
            user_id = query_filter.must[0].match.value
            matches = [point for point in matches if point.payload["user_id"] == user_id]
        return SimpleNamespace(points=matches[:limit])

    def scroll(self, collection_name, scroll_filter=None, limit=1000, with_payload=True, with_vectors=False):
        points = list(self.points)
        if scroll_filter is not None:
            must = scroll_filter.must
            user_id = must[0].match.value
            pair_id = must[1].match.value if len(must) > 1 else None
            points = []
            for point in self.points:
                payload = point.payload
                if payload["user_id"] != user_id:
                    continue
                if pair_id is not None and payload.get("pair_id") != pair_id:
                    continue
                points.append(point)
        return points[:limit], None

    def delete(self, collection_name, points_selector):
        doomed = set(points_selector.points)
        self.points = [p for p in self.points if p.id not in doomed]

    def set_payload(self, collection_name, payload, points):
        doomed = set(points)
        for point in self.points:
            if point.id in doomed:
                point.payload.update(payload)


def make_memory(monkeypatch, user_id="user_1"):
    MemoryOS._user_pairs.clear()
    clients = []
    embedders = []

    def client_factory(*args, **kwargs):
        client = FakeQdrantClient(*args, **kwargs)
        clients.append(client)
        return client

    def embedder_factory(*args, **kwargs):
        embedder = FakeSentenceTransformer()
        embedders.append(embedder)
        return embedder

    monkeypatch.setattr("memory_os.memory.QdrantClient", client_factory)
    monkeypatch.setattr("memory_os.memory.SentenceTransformer", embedder_factory)
    memory = MemoryOS(
        qdrant_url="http://qdrant.test",
        qdrant_api_key="key",
        llm=FakeLLM(),
    )
    return memory, clients[0], embedders[0]


def test_store_extracts_multiple_memories_with_metadata(monkeypatch):
    memory, client, embedder = make_memory(monkeypatch)

    memory.store("I prefer Python and I am building an auth feature", "Got it", user_id="user_1")

    assert len(memory.llm.prompts) == 1
    assert embedder.texts == [
        "User prefers Python",
        "User is building an auth feature",
    ]
    assert len(client.points) == 2
    first_payload = client.points[0].payload
    assert first_payload["text"] == "User prefers Python"
    assert first_payload["importance"] == 0.95
    assert first_payload["emotion"] == "joy"
    assert first_payload["emotional_weight"] > 1.0
    assert first_payload["decay_score"] == 1.0
    assert ("user_id", "keyword") in client.indexed_fields
    assert ("pair_id", "keyword") in client.indexed_fields


def test_retrieve_updates_decay_state_and_returns_scores(monkeypatch):
    memory, client, _ = make_memory(monkeypatch)

    memory.store("I have a deadline and I am stressed", "Stored", user_id="user_1")
    point = client.points[0]
    point.payload["last_accessed"] = point.payload["created_at"] - (2 * 86400)

    results = memory.retrieve("deadline", user_id="user_1")

    assert results["memories"][0]["text"] == "User is stressed about a launch deadline"
    assert results["memories"][0]["score"] is not None
    assert results["memories"][0]["decay_score"] < 1.0
    assert point.payload["access_count"] == 1
    assert point.payload["final_score"] == results["memories"][0]["score"]


def test_retrieve_is_scoped_by_user_and_returns_recent_pairs(monkeypatch):
    memory, client, embedder = make_memory(monkeypatch)
    other, _, _ = make_memory(monkeypatch)
    other._client = client
    other._collection_ready = True

    memory.store("I prefer Python and I am building an auth feature", "Stored", user_id="user_1")
    other.store("I have a deadline and I am stressed", "Stored", user_id="user_2")

    results = memory.retrieve("What do I prefer?", user_id="user_1")

    assert [item["text"] for item in results["memories"]] == [
        "User prefers Python",
        "User is building an auth feature",
    ]
    assert results["recent_pairs"] == [
        {
            "prompt": "I prefer Python and I am building an auth feature",
            "response": "Stored",
            "created_at": MemoryOS._user_pairs["user_1"][0]["created_at"],
        }
    ]
    assert embedder.texts[-1] == "What do I prefer?"


def test_store_keeps_only_latest_seven_pairs(monkeypatch):
    memory, client, _ = make_memory(monkeypatch)

    for index in range(8):
        memory.store(f"prompt {index}", f"response {index}", user_id="user_1")

    assert len(MemoryOS._user_pairs["user_1"]) == 7
    assert [pair["prompt"] for pair in MemoryOS._user_pairs["user_1"]] == [
        "prompt 1",
        "prompt 2",
        "prompt 3",
        "prompt 4",
        "prompt 5",
        "prompt 6",
        "prompt 7",
    ]
    assert len(client.points) == 7


def test_retrieve_returns_latest_five_prompt_response_pairs(monkeypatch):
    memory, _, _ = make_memory(monkeypatch)

    for index in range(7):
        memory.store(f"prompt {index}", f"response {index}", user_id="user_1")

    results = memory.retrieve("new prompt", user_id="user_1")
    recent_pairs = results["recent_pairs"]

    assert len(recent_pairs) == 5
    assert [pair["prompt"] for pair in recent_pairs] == [
        "prompt 2",
        "prompt 3",
        "prompt 4",
        "prompt 5",
        "prompt 6",
    ]


def test_consolidate_merges_cluster_and_leaves_singletons(monkeypatch):
    memory, client, _ = make_memory(monkeypatch)

    cluster_a = PointStruct(
        id="merged-a",
        vector=[0.2, 0.2, 0.2],
        payload={
            "user_id": "user_1",
            "pair_id": "pair-1",
            "text": "The user likes Python.",
            "created_at": 1.0,
            "last_accessed": 1.0,
            "access_count": 2,
            "stability": 2.0,
            "importance": 0.8,
            "emotion": "joy",
            "emotional_weight": 1.15,
            "decay_score": 1.0,
            "final_score": 0.92,
        },
    )
    cluster_b = PointStruct(
        id="merged-b",
        vector=[0.2, 0.2, 0.2],
        payload={
            "user_id": "user_1",
            "pair_id": "pair-1",
            "text": "The user likes Python for automation.",
            "created_at": 2.0,
            "last_accessed": 2.0,
            "access_count": 1,
            "stability": 2.0,
            "importance": 0.7,
            "emotion": "joy",
            "emotional_weight": 1.15,
            "decay_score": 1.0,
            "final_score": 0.805,
        },
    )
    singleton = PointStruct(
        id="singleton-a",
        vector=[1.0, 0.0, 0.0],
        payload={
            "user_id": "user_1",
            "pair_id": "pair-2",
            "text": "The user prefers Rust.",
            "created_at": 3.0,
            "last_accessed": 3.0,
            "access_count": 1,
            "stability": 2.0,
            "importance": 0.9,
            "emotion": "sadness",
            "emotional_weight": 1.2,
            "decay_score": 1.0,
            "final_score": 1.08,
        },
    )
    client.points = [cluster_a, cluster_b, singleton]

    summary = memory.consolidate("user_1")

    assert summary == {"clusters_merged": 1, "points_removed": 2}
    assert len([point for point in client.points if point.payload["text"] == "The user prefers Rust."]) == 1
    assert len([point for point in client.points if point.payload["text"] == "Merged memory text preserving the distinct facts"]) == 1


def test_consolidate_singletons_are_not_merged(monkeypatch):
    memory, client, _ = make_memory(monkeypatch)

    singleton_one = PointStruct(
        id="single-one",
        vector=[0.7, 0.0, 0.0],
        payload={
            "user_id": "user_1",
            "pair_id": "pair-1",
            "text": "The user likes Python.",
            "created_at": 1.0,
            "last_accessed": 1.0,
            "access_count": 2,
            "stability": 2.0,
            "importance": 0.8,
            "emotion": "joy",
            "emotional_weight": 1.15,
            "decay_score": 1.0,
            "final_score": 0.92,
        },
    )
    singleton_two = PointStruct(
        id="single-two",
        vector=[0.0, 0.7, 0.0],
        payload={
            "user_id": "user_1",
            "pair_id": "pair-2",
            "text": "The user likes Rust.",
            "created_at": 2.0,
            "last_accessed": 2.0,
            "access_count": 1,
            "stability": 2.0,
            "importance": 0.7,
            "emotion": "neutral",
            "emotional_weight": 1.0,
            "decay_score": 1.0,
            "final_score": 0.7,
        },
    )
    client.points = [singleton_one, singleton_two]

    summary = memory.consolidate("user_1")

    assert summary == {"clusters_merged": 0, "points_removed": 0}
    assert len(client.points) == 2
    assert sorted(point.id for point in client.points) == ["single-one", "single-two"]


def test_consolidate_merged_points_do_not_get_deleted_by_pair_eviction(monkeypatch):
    memory, client, _ = make_memory(monkeypatch)

    a = PointStruct(
        id="pair-delete-a",
        vector=[0.2, 0.2, 0.2],
        payload={
            "user_id": "user_1",
            "pair_id": "pair-delete",
            "text": "The user likes Python.",
            "created_at": 1.0,
            "last_accessed": 1.0,
            "access_count": 2,
            "stability": 2.0,
            "importance": 0.8,
            "emotion": "joy",
            "emotional_weight": 1.15,
            "decay_score": 1.0,
            "final_score": 0.92,
        },
    )
    b = PointStruct(
        id="pair-delete-b",
        vector=[0.2, 0.2, 0.2],
        payload={
            "user_id": "user_1",
            "pair_id": "pair-delete",
            "text": "The user likes Python for automation.",
            "created_at": 2.0,
            "last_accessed": 2.0,
            "access_count": 1,
            "stability": 2.0,
            "importance": 0.7,
            "emotion": "joy",
            "emotional_weight": 1.15,
            "decay_score": 1.0,
            "final_score": 0.805,
        },
    )
    client.points = [a, b]

    memory.consolidate("user_1")
    memory._delete_pair_memories("user_1", "pair-delete")

    merged_point = [point for point in client.points if point.payload.get("merged_from") == 2]
    assert len(merged_point) == 1
    assert merged_point[0].payload["pair_id"] is None


def test_store_consolidation_recommended_flag_threshold_and_reset(monkeypatch):
    memory, client, _ = make_memory(monkeypatch)

    for index in range(1, 20):
        result = memory.store(f"prompt {index}", f"response {index}", user_id="user_1")
        assert result == {"consolidation_recommended": False}

    result_20 = memory.store("prompt 20", "response 20", user_id="user_1")
    assert result_20 == {"consolidation_recommended": True}

    result_21 = memory.store("prompt 21", "response 21", user_id="user_1")
    assert result_21 == {"consolidation_recommended": False}

    result_user2 = memory.store("prompt user 2", "response user 2", user_id="user_2")
    assert result_user2 == {"consolidation_recommended": False}
    assert memory._store_counts.get("user_2") == 1


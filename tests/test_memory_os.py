import json
import time
from types import SimpleNamespace

from memory_os.memory import MemoryOS


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
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
            session_id = query_filter.must[0].match.value
            matches = [point for point in matches if point.payload["session_id"] == session_id]
        return SimpleNamespace(points=matches[:limit])

    def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
        points = self.points
        if scroll_filter is not None:
            must = scroll_filter.must
            session_id = must[0].match.value
            pair_id = must[1].match.value if len(must) > 1 else None
            points = []
            for point in self.points:
                payload = point.payload
                if payload["session_id"] != session_id:
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


def make_memory(monkeypatch, session_id="user_1"):
    MemoryOS._session_pairs.clear()
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
        session_id=session_id,
    )
    return memory, clients[0], embedders[0]


def test_store_extracts_multiple_memories_with_metadata(monkeypatch):
    memory, client, embedder = make_memory(monkeypatch)

    memory.store("I prefer Python and I am building an auth feature", "Got it")

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
    assert ("session_id", "keyword") in client.indexed_fields
    assert ("pair_id", "keyword") in client.indexed_fields


def test_retrieve_updates_decay_state_and_returns_scores(monkeypatch):
    memory, client, _ = make_memory(monkeypatch)

    memory.store("I have a deadline and I am stressed", "Stored")
    point = client.points[0]
    point.payload["last_accessed"] = point.payload["created_at"] - (2 * 86400)

    results = memory.retrieve("deadline")

    assert results["memories"][0]["text"] == "User is stressed about a launch deadline"
    assert results["memories"][0]["score"] is not None
    assert results["memories"][0]["decay_score"] < 1.0
    assert point.payload["access_count"] == 1
    assert point.payload["final_score"] == results["memories"][0]["score"]


def test_retrieve_is_scoped_by_session_and_returns_recent_pairs(monkeypatch):
    memory, client, embedder = make_memory(monkeypatch, session_id="user_1")
    other, _, _ = make_memory(monkeypatch, session_id="user_2")
    other._client = client
    other._collection_ready = True

    memory.store("I prefer Python and I am building an auth feature", "Stored")
    other.store("I have a deadline and I am stressed", "Stored")

    results = memory.retrieve("What do I prefer?")

    assert [item["text"] for item in results["memories"]] == [
        "User prefers Python",
        "User is building an auth feature",
    ]
    assert results["recent_pairs"] == [
        {
            "prompt": "I prefer Python and I am building an auth feature",
            "response": "Stored",
            "created_at": MemoryOS._session_pairs["user_1"][0]["created_at"],
        }
    ]
    assert embedder.texts[-1] == "What do I prefer?"


def test_store_keeps_only_latest_seven_pairs(monkeypatch):
    memory, client, _ = make_memory(monkeypatch)

    for index in range(8):
        memory.store(f"prompt {index}", f"response {index}")

    assert len(MemoryOS._session_pairs["user_1"]) == 7
    assert [pair["prompt"] for pair in MemoryOS._session_pairs["user_1"]] == [
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
        memory.store(f"prompt {index}", f"response {index}")

    results = memory.retrieve("new prompt")
    recent_pairs = results["recent_pairs"]

    assert len(recent_pairs) == 5
    assert [pair["prompt"] for pair in recent_pairs] == [
        "prompt 2",
        "prompt 3",
        "prompt 4",
        "prompt 5",
        "prompt 6",
    ]

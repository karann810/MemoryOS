from pytest import MonkeyPatch

from memory_os.memory import MemoryOS, PointStruct
from tests.test_memory_os import FakeLLM, FakeQdrantClient, FakeSentenceTransformer

mp = MonkeyPatch()
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

mp.setattr('memory_os.memory.QdrantClient', client_factory)
mp.setattr('memory_os.memory.SentenceTransformer', embedder_factory)
mp.setattr('memory_os.memory.AgglomerativeClustering', __import__('sklearn.cluster').cluster.AgglomerativeClustering)

memory = MemoryOS(qdrant_url='http://qdrant.test', qdrant_api_key='key', llm=FakeLLM())
client = clients[0]

cluster_a = PointStruct(id='merged-a', vector=[0.2,0.2,0.2], payload={'user_id':'user_1','pair_id':'pair-1','text':'The user likes Python.','created_at':1.0,'last_accessed':1.0,'access_count':2,'stability':2.0,'importance':0.8,'emotion':'joy','emotional_weight':1.15,'decay_score':1.0,'final_score':0.92})
cluster_b = PointStruct(id='merged-b', vector=[0.2,0.2,0.2], payload={'user_id':'user_1','pair_id':'pair-1','text':'The user likes Python for automation.','created_at':2.0,'last_accessed':2.0,'access_count':1,'stability':2.0,'importance':0.7,'emotion':'joy','emotional_weight':1.15,'decay_score':1.0,'final_score':0.805})
singleton = PointStruct(id='singleton-a', vector=[1.0,0.0,0.0], payload={'user_id':'user_1','pair_id':'pair-2','text':'The user prefers Rust.','created_at':3.0,'last_accessed':3.0,'access_count':1,'stability':2.0,'importance':0.9,'emotion':'sadness','emotional_weight':1.2,'decay_score':1.0,'final_score':1.08})
client.points = [cluster_a, cluster_b, singleton]

summary = memory.consolidate('user_1')
print(summary)
print('len points', len(client.points))
for p in client.points:
    print(p.id, p.payload.get('text'), p.payload.get('merged_from'), p.payload.get('pair_id'))
print('prompts', memory.llm.prompts)
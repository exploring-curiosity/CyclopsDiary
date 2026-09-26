"""ElideDB — a Parquet-only, timestamp-first multimodal store.

>>> from elidedb import Store
>>> db = Store.open("lake/oxford")
>>> db.describe()
>>> data, stats = db.window(t0, t1)
>>> db.sql("SELECT count(*) FROM gps")
>>> db.search_text("pedestrians crossing")
"""
from .store import Store, Table, QueryStats
from .embeddings import cluster, embed_text

__version__ = "2.0.0"
__all__ = ["Store", "Table", "QueryStats", "cluster", "embed_text"]

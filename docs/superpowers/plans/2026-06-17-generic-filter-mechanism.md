# Generic Filter Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `topic_id` parameter across the SDK with a generic `filters: dict[str, Any]` mechanism, making the SDK domain-agnostic and extensible.

**Architecture:** Replace `topic_id: str | None` with `filters: dict[str, Any] | None` at every layer — from the `DatabaseBackend` protocol through both implementations (`AsyncPgBackend`, `SyncPgBackend`), the DDL/DQL SQL templates, the processors (`RetrieveProcessor`, `IngestProcessor`), and the contracts. The SQL layer dynamically builds parameterized WHERE clauses from the filters dict. Ingest gains a matching `article_columns: dict[str, Any] | None` parameter to explicitly set article-level columns during upsert (replacing the implicit `metadata.get("topic_id")` pattern).

**Tech Stack:** Python 3.10+, SQLAlchemy (raw `text()` SQL), asyncpg/psycopg2, Pydantic (contracts), pytest + pytest-asyncio

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/chatbot_plugin_sdk/backends/base.py` | Protocol: `topic_id` → `filters`; `SearchRow`: remove `topic_id` field |
| Modify | `src/chatbot_plugin_sdk/backends/_pg_ddl.py` | DML/DQL: dynamic WHERE/INSERT column generation from filters/article_columns |
| Modify | `src/chatbot_plugin_sdk/backends/async_pg.py` | AsyncPgBackend: adopt `filters` param, dynamic SQL |
| Modify | `src/chatbot_plugin_sdk/backends/sync_pg.py` | SyncPgBackend: adopt `filters` param, dynamic SQL |
| Modify | `src/chatbot_plugin_sdk/processors/retrieve.py` | RetrieveProcessor: `topic_id` → `filters` |
| Modify | `src/chatbot_plugin_sdk/processors/ingest.py` | IngestProcessor: add `article_columns` param to `ingest()` |
| Modify | `src/chatbot_plugin_sdk/contracts/responses.py` | ChunkResult: no changes needed (already has no topic_id) |
| Modify | `src/chatbot_plugin_sdk/contracts/requests.py` | SearchRequest: add `filters` field |
| Modify | `src/chatbot_plugin_sdk/models/article.py` | Article ORM: add `public_article_id`, `topic_id` columns (was missing) |
| Modify | `src/tests/test_retrieve.py` | Update tests for `filters` parameter |
| Modify | `src/tests/test_ingest.py` | Update tests for `article_columns` parameter |

---

## Key Design Decisions

### 1. `filters` dict semantics (search/retrieve side)

```python
filters: dict[str, Any] | None = None
# Example: {"topic_id": "uuid-xxx", "source": "wiki"}
```

- Keys **must** match actual column names on the `articles` table.
- All filter conditions are AND-ed together.
- Values are parameterized — **never** interpolated into SQL strings (SQL injection safety).
- `None` or empty dict means "no filter" (same as current `topic_id=None`).

### 2. `article_columns` dict semantics (ingest/upsert side)

```python
article_columns: dict[str, Any] | None = None
# Example: {"topic_id": "uuid-xxx", "source": "internal"}
```

- Keys that correspond to real table columns are extracted and placed in the INSERT/UPDATE.
- Remaining keys (or keys not matching known columns) go into the JSONB `metadata` column.
- This replaces the implicit `metadata.get("topic_id")` pattern — callers explicitly declare which keys are structural columns.

### 3. Known article columns (the "structural" set)

The following keys are recognized as real columns on the `articles` table (not JSONB):
- `url` (required, always from metadata)
- `title`
- `source`
- `public_article_id`
- `topic_id`

The DDL already defines these. The `_pg_ddl.py` module will export a constant `_ARTICLE_COLUMNS` listing them, and a helper `_split_article_fields(metadata, article_columns)` that returns `(col_params, jsonb_metadata)`.

### 4. Backward compatibility

This is a **breaking change**: `topic_id` parameter is removed from public APIs. The SDK has no stable release yet, so breaking changes are acceptable. A deprecation shim is **not** included — callers must update to use `filters` / `article_columns`.

### 5. SQL generation safety

Dynamic WHERE clauses are built from a whitelist of known column names (`_ARTICLE_COLUMNS`). Any key in `filters` that is **not** in the whitelist raises `DatabaseError` at query time — this prevents both typos and injection.

---

### Task 1: Define constants and helpers in `_pg_ddl.py`

**Files:**
- Modify: `src/chatbot_plugin_sdk/backends/_pg_ddl.py`

This task establishes the foundation: the known-column whitelist and the helper that splits article fields.

- [ ] **Step 1: Add `_ARTICLE_COLUMNS` constant and `_split_article_fields` helper**

Add after the existing imports and before the DDL constants:

```python
_ARTICLE_COLUMNS = frozenset({
    "url", "title", "source", "public_article_id", "topic_id",
})


def _split_article_fields(
    metadata: dict,
    article_columns: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict | None]:
    """Split article-level fields from metadata into SQL column params vs JSONB blob.

    Args:
        metadata:         Caller-supplied metadata dict (must contain 'url').
        article_columns:  Explicit column values to write into article table columns.
                          Keys must be in _ARTICLE_COLUMNS.

    Returns:
        (col_params, jsonb_metadata) where:
          - col_params maps column names to values for the INSERT/UPDATE
          - jsonb_metadata is the remaining metadata to store as JSONB
    """
    col_params: dict[str, Any] = {}
    # Always pull the core fields from metadata for backward compatibility
    for key in ("url", "title", "source", "public_article_id"):
        if key in metadata:
            col_params[key] = metadata[key]

    # Explicit article_columns override / extend
    if article_columns:
        for key, value in article_columns.items():
            if key not in _ARTICLE_COLUMNS:
                raise DatabaseError(
                    f"article_columns key {key!r} is not a known article column. "
                    f"Known columns: {sorted(_ARTICLE_COLUMNS)}"
                )
            col_params[key] = value

    # Build JSONB metadata: everything in metadata that is NOT a structural column
    jsonb_keys = set(metadata.keys()) - _ARTICLE_COLUMNS
    jsonb_metadata = {k: metadata[k] for k in jsonb_keys if k in metadata} or None

    return col_params, jsonb_metadata
```

Also add the `Any` import at the top of the file:

```python
from typing import Any
```

- [ ] **Step 2: Replace hardcoded `_DML_UPSERT_ARTICLE` with a builder function**

The current DML hardcodes every column. Replace it with a function that generates the SQL dynamically from `col_params` keys:

```python
def _build_upsert_article_sql(
    schema: str,
    articles_table: str,
    col_params: dict[str, Any],
) -> str:
    """Build parameterized INSERT ... ON CONFLICT UPDATE for the articles table."""
    cols = sorted(col_params.keys())
    # 'id' is always first, metadata always last
    insert_cols = ["id"] + cols + ["metadata"]
    insert_placeholders = [f":{c}" for c in insert_cols]
    # CAST UUID columns
    uuid_cols = {"id", "public_article_id", "topic_id"}
    cast_placeholders = []
    for c in insert_cols:
        if c in uuid_cols:
            cast_placeholders.append(f"CAST(:{c} AS UUID)")
        elif c == "metadata":
            cast_placeholders.append("CAST(:metadata AS JSONB)")
        else:
            cast_placeholders.append(f":{c}")

    update_sets = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in cols if c != "url"
    ) + ", metadata = EXCLUDED.metadata, updated_at = now()"

    return (
        f"INSERT INTO {schema}.{articles_table} ({', '.join(insert_cols)})\n"
        f"VALUES ({', '.join(cast_placeholders)})\n"
        f"ON CONFLICT (id) DO UPDATE SET\n    {update_sets}"
    )
```

- [ ] **Step 3: Replace hardcoded `_DQL_SEARCH_DENSE` and `_DQL_SEARCH_SPARSE` with builder functions**

Replace the WHERE clause `AND (:topic_id IS NULL OR a.topic_id = CAST(:topic_id AS UUID))` with dynamic filter generation:

```python
def _build_search_where(
    filters: dict[str, Any] | None,
    table_alias: str = "a",
) -> tuple[str, dict[str, Any]]:
    """Build parameterized WHERE clause fragment from filters dict.

    Returns:
        (where_fragment, params) — the fragment starts with "AND ..."
        and params contains the bound values.
    """
    if not filters:
        return "", {}

    uuid_cols = {"public_article_id", "topic_id"}
    fragments = []
    params: dict[str, Any] = {}

    for col, value in filters.items():
        if col not in _ARTICLE_COLUMNS:
            raise DatabaseError(
                f"filter key {col!r} is not a known article column. "
                f"Known columns: {sorted(_ARTICLE_COLUMNS)}"
            )
        param_name = f"_f_{col}"
        if col in uuid_cols:
            fragments.append(
                f"({table_alias}.{col} = CAST(:{param_name} AS UUID))"
            )
        else:
            fragments.append(
                f"({table_alias}.{col} = :{param_name})"
            )
        params[param_name] = value

    where = " AND " + " AND ".join(fragments)
    return where, params


def _build_search_dense_sql(
    schema: str,
    articles_table: str,
    chunks_table: str,
    filters: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    where_frag, filter_params = _build_search_where(filters)
    sql = (
        f"SELECT\n"
        f"    ac.id                    AS chunk_id,\n"
        f"    ac.article_id,\n"
        f"    ac.chunk_index,\n"
        f"    ac.content,\n"
        f"    a.title,\n"
        f"    a.url,\n"
        f"    a.public_article_id,\n"
        f"    ac.dense_vector <=> CAST(:query_vec AS vector) AS distance\n"
        f"FROM {schema}.{chunks_table} ac\n"
        f"JOIN {schema}.{articles_table} a ON ac.article_id = a.id\n"
        f"WHERE ac.dense_vector IS NOT NULL\n"
        f"{where_frag}\n"
        f"ORDER BY distance\n"
        f"LIMIT :top_k"
    )
    return sql, filter_params


def _build_search_sparse_sql(
    schema: str,
    articles_table: str,
    chunks_table: str,
    filters: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    where_frag, filter_params = _build_search_where(filters)
    sql = (
        f"SELECT\n"
        f"    ac.id                    AS chunk_id,\n"
        f"    ac.article_id,\n"
        f"    ac.chunk_index,\n"
        f"    ac.content,\n"
        f"    a.title,\n"
        f"    a.url,\n"
        f"    a.public_article_id,\n"
        f"    (ac.sparse_vector <#> CAST(:query_vec AS sparsevec)) AS distance\n"
        f"FROM {schema}.{chunks_table} ac\n"
        f"JOIN {schema}.{articles_table} a ON ac.article_id = a.id\n"
        f"WHERE ac.sparse_vector IS NOT NULL\n"
        f"{where_frag}\n"
        f"ORDER BY distance\n"
        f"LIMIT :top_k"
    )
    return sql, filter_params
```

Note: `a.topic_id` is removed from the SELECT clause — `SearchRow` will no longer carry `topic_id`. If callers need it, they can reconstruct it from `article_id`.

- [ ] **Step 4: Remove old constants that are no longer used**

Delete `_DML_UPSERT_ARTICLE`, `_DQL_SEARCH_DENSE`, and `_DQL_SEARCH_SPARSE` from the file (they are replaced by the builder functions above). Keep `_DDL_CREATE_ARTICLES`, `_DDL_CREATE_CHUNKS`, and all other DDL/DML constants as-is (the DDL for `topic_id` column stays — it's a valid column, just not hardcoded in queries anymore).

- [ ] **Step 5: Commit**

```bash
git add src/chatbot_plugin_sdk/backends/_pg_ddl.py
git commit -m "refactor: replace hardcoded topic_id SQL with generic filter/column builders"
```

---

### Task 2: Update `DatabaseBackend` protocol and `SearchRow`

**Files:**
- Modify: `src/chatbot_plugin_sdk/backends/base.py`

- [ ] **Step 1: Write the failing test**

Add to `src/tests/test_retrieve.py`:

```python
class TestFilterParameter:
    @pytest.mark.asyncio
    async def test_retrieve_accepts_filters_dict(self):
        """retrieve() should accept a filters dict instead of topic_id."""
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = []
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            result = await retriever.retrieve("q", filters={"topic_id": "abc-123"})

        # Should pass filters dict to backend
        call_kwargs = backend.search_dense.call_args.kwargs
        assert call_kwargs.get("filters") == {"topic_id": "abc-123"}
        assert isinstance(result, SearchResponse)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && python -m pytest src/tests/test_retrieve.py::TestFilterParameter::test_retrieve_accepts_filters_dict -v`
Expected: FAIL — `retrieve()` does not accept `filters` keyword yet.

- [ ] **Step 3: Update `SearchRow` — remove `topic_id` field**

In `src/chatbot_plugin_sdk/backends/base.py`, change `SearchRow`:

```python
@dataclass
class SearchRow:
    """Single search result row returned by DatabaseBackend.search_dense()."""
    chunk_id: str
    article_id: str
    chunk_index: int
    content: str
    title: str | None
    url: str | None
    distance: float
    public_article_id: str | None = None
```

- [ ] **Step 4: Update `DatabaseBackend` protocol — replace `topic_id` with `filters`**

In `src/chatbot_plugin_sdk/backends/base.py`, update the protocol:

```python
async def search_dense(
    self,
    query_vec: list[float],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[SearchRow]:
    """Cosine similarity search on the dense_vector column.

    Args:
        filters: Column-level filters on the articles table, e.g.
                 ``{"topic_id": "uuid"}``. Keys must be known article columns.
    """
    ...

async def search_sparse(
    self,
    query_vec: dict[str, float],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[SearchRow]:
    """Maximum inner product search on the sparse_vector column (<#> operator)."""
    ...
```

Add `from typing import Any` to the imports if not already present.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && python -m pytest src/tests/test_retrieve.py::TestFilterParameter::test_retrieve_accepts_filters_dict -v`
Expected: FAIL — `RetrieveProcessor` still passes `topic_id`. Need to update processor next. But the protocol change is validated.

- [ ] **Step 6: Commit**

```bash
git add src/chatbot_plugin_sdk/backends/base.py src/tests/test_retrieve.py
git commit -m "refactor: replace topic_id with filters in DatabaseBackend protocol and SearchRow"
```

---

### Task 3: Update `AsyncPgBackend` implementation

**Files:**
- Modify: `src/chatbot_plugin_sdk/backends/async_pg.py`

- [ ] **Step 1: Update `search_dense` signature and implementation**

Replace the `search_dense` method:

```python
async def search_dense(
    self,
    query_vec: list[float],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[SearchRow]:
    from chatbot_plugin_sdk.backends._pg_ddl import _build_search_dense_sql
    schema = self.schema
    at = self.articles_table
    ct = self.chunks_table
    sql, filter_params = _build_search_dense_sql(schema, at, ct, filters)
    params = {"query_vec": _dense_vec_str(query_vec), "top_k": top_k}
    params.update(filter_params)
    async with self._engine.connect() as conn:
        rows = (await conn.execute(text(sql), params)).all()
    return [
        SearchRow(
            chunk_id=str(r.chunk_id),
            article_id=str(r.article_id),
            chunk_index=r.chunk_index,
            content=r.content,
            title=r.title,
            url=r.url,
            distance=float(r.distance),
            public_article_id=str(r.public_article_id) if r.public_article_id else None,
        )
        for r in rows
    ]
```

- [ ] **Step 2: Update `search_sparse` signature and implementation**

Replace the `search_sparse` method:

```python
async def search_sparse(
    self,
    query_vec: dict[str, float],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[SearchRow]:
    from chatbot_plugin_sdk.backends._pg_ddl import _build_search_sparse_sql
    if not self._sparse_dim:
        return []
    schema = self.schema
    at = self.articles_table
    ct = self.chunks_table
    sql, filter_params = _build_search_sparse_sql(schema, at, ct, filters)
    sv_str = _to_sparsevec_string(query_vec, self._sparse_dim)
    params = {"query_vec": sv_str, "top_k": top_k}
    params.update(filter_params)
    async with self._engine.connect() as conn:
        rows = (await conn.execute(text(sql), params)).all()
    return [
        SearchRow(
            chunk_id=str(r.chunk_id),
            article_id=str(r.article_id),
            chunk_index=r.chunk_index,
            content=r.content,
            title=r.title,
            url=r.url,
            distance=float(r.distance),
            public_article_id=str(r.public_article_id) if r.public_article_id else None,
        )
        for r in rows
    ]
```

- [ ] **Step 3: Update `upsert` method to use `_split_article_fields` and `_build_upsert_article_sql`**

Replace the upsert method:

```python
async def upsert(
    self,
    article_id: uuid.UUID,
    metadata: dict,
    chunks: list[str],
    dense_vectors: list[list[float]] | None,
    sparse_vectors: list[dict[str, float]] | None,
    article_columns: dict[str, Any] | None = None,
) -> None:
    from chatbot_plugin_sdk.backends._pg_ddl import (
        _build_upsert_article_sql,
        _split_article_fields,
    )
    schema = self.schema
    at = self.articles_table
    ct = self.chunks_table
    col_params, jsonb_metadata = _split_article_fields(metadata, article_columns)
    try:
        async with self._engine.begin() as conn:
            sql = _build_upsert_article_sql(schema, at, col_params)
            params = {"id": str(article_id)}
            params.update(col_params)
            params["metadata"] = json.dumps(jsonb_metadata) if jsonb_metadata is not None else None
            await conn.execute(text(sql), params)

            await conn.execute(
                text(_DML_DELETE_CHUNKS.format(schema=schema, chunks_table=ct)),
                {"article_id": str(article_id)},
            )
            for i, content in enumerate(chunks):
                dense_val = _dense_vec_str(dense_vectors[i]) if dense_vectors else None
                sparse_val = (
                    _to_sparsevec_string(sparse_vectors[i], self._sparse_dim)
                    if sparse_vectors is not None and self._sparse_dim
                    else None
                )
                await conn.execute(
                    text(_DML_INSERT_CHUNK.format(schema=schema, chunks_table=ct)),
                    {
                        "article_id": str(article_id),
                        "chunk_index": i,
                        "content": content,
                        "dense_vector": dense_val,
                        "sparse_vector": sparse_val,
                    },
                )
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Upsert failed for article {article_id}: {exc}") from exc
```

- [ ] **Step 4: Update imports — remove old DQL/DML constants, add new builders**

Update the import block at top of `async_pg.py`:

```python
from chatbot_plugin_sdk.backends._pg_ddl import (
    _build_search_dense_sql,
    _build_search_sparse_sql,
    _build_upsert_article_sql,
    _DDL_CREATE_CHUNKS,
    _DDL_CREATE_ARTICLES,
    _DDL_CREATE_EXTENSION,
    _DDL_CREATE_SCHEMA,
    _DDL_IDX_SOURCE,
    _DDL_IDX_URL,
    _DDL_TABLE_EXISTS,
    _DML_DELETE_CHUNKS,
    _DML_INSERT_CHUNK,
    _check_dim_from_cols,
    _check_sparse_dim_from_cols,
    _dense_col_ddl,
    _dense_vec_str,
    _sparse_col_ddl,
    _to_sparsevec_string,
)
```

Note: `_DML_UPSERT_ARTICLE`, `_DQL_SEARCH_DENSE`, `_DQL_SEARCH_SPARSE` are removed (replaced by builder functions). But `_build_search_dense_sql` etc. are imported lazily inside the methods to avoid circular imports — so actually import them at the top level instead:

Actually, on reflection, keep the imports at the top (no circular import risk — `_pg_ddl.py` does not import from `async_pg.py`). Remove the lazy `from` imports inside method bodies and put them at the top.

- [ ] **Step 5: Commit**

```bash
git add src/chatbot_plugin_sdk/backends/async_pg.py
git commit -m "refactor: AsyncPgBackend uses generic filters and article_columns"
```

---

### Task 4: Update `SyncPgBackend` implementation

**Files:**
- Modify: `src/chatbot_plugin_sdk/backends/sync_pg.py`

This mirrors the changes from Task 3 for the synchronous backend.

- [ ] **Step 1: Update async wrapper signatures**

```python
async def search_dense(
    self,
    query_vec: list[float],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[SearchRow]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, self._search_dense_sync, query_vec, top_k, filters)

async def search_sparse(
    self,
    query_vec: dict[str, float],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[SearchRow]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, self._search_sparse_sync, query_vec, top_k, filters)
```

Update `upsert` wrapper:

```python
async def upsert(
    self,
    article_id: uuid.UUID,
    metadata: dict,
    chunks: list[str],
    dense_vectors: list[list[float]] | None,
    sparse_vectors: list[dict[str, float]] | None,
    article_columns: dict[str, Any] | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, self._upsert_sync,
        article_id, metadata, chunks, dense_vectors, sparse_vectors, article_columns,
    )
```

- [ ] **Step 2: Update sync implementations**

Replace `_search_dense_sync`:

```python
def _search_dense_sync(
    self,
    query_vec: list[float],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[SearchRow]:
    schema = self.schema
    at = self.articles_table
    ct = self.chunks_table
    sql, filter_params = _build_search_dense_sql(schema, at, ct, filters)
    params = {"query_vec": _dense_vec_str(query_vec), "top_k": top_k}
    params.update(filter_params)
    with self._engine.connect() as conn:
        rows = conn.execute(text(sql), params).all()
    return [
        SearchRow(
            chunk_id=str(r.chunk_id),
            article_id=str(r.article_id),
            chunk_index=r.chunk_index,
            content=r.content,
            title=r.title,
            url=r.url,
            distance=float(r.distance),
            public_article_id=str(r.public_article_id) if r.public_article_id else None,
        )
        for r in rows
    ]
```

Replace `_search_sparse_sync`:

```python
def _search_sparse_sync(
    self,
    query_vec: dict[str, float],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[SearchRow]:
    if not self._sparse_dim:
        return []
    schema = self.schema
    at = self.articles_table
    ct = self.chunks_table
    sql, filter_params = _build_search_sparse_sql(schema, at, ct, filters)
    sv_str = _to_sparsevec_string(query_vec, self._sparse_dim)
    params = {"query_vec": sv_str, "top_k": top_k}
    params.update(filter_params)
    with self._engine.connect() as conn:
        rows = conn.execute(text(sql), params).all()
    return [
        SearchRow(
            chunk_id=str(r.chunk_id),
            article_id=str(r.article_id),
            chunk_index=r.chunk_index,
            content=r.content,
            title=r.title,
            url=r.url,
            distance=float(r.distance),
            public_article_id=str(r.public_article_id) if r.public_article_id else None,
        )
        for r in rows
    ]
```

Replace `_upsert_sync`:

```python
def _upsert_sync(
    self,
    article_id: uuid.UUID,
    metadata: dict,
    chunks: list[str],
    dense_vectors: list[list[float]] | None,
    sparse_vectors: list[dict[str, float]] | None,
    article_columns: dict[str, Any] | None = None,
) -> None:
    schema = self.schema
    at = self.articles_table
    ct = self.chunks_table
    col_params, jsonb_metadata = _split_article_fields(metadata, article_columns)
    try:
        with self._engine.begin() as conn:
            sql = _build_upsert_article_sql(schema, at, col_params)
            params = {"id": str(article_id)}
            params.update(col_params)
            params["metadata"] = json.dumps(jsonb_metadata) if jsonb_metadata is not None else None
            conn.execute(text(sql), params)

            conn.execute(
                text(_DML_DELETE_CHUNKS.format(schema=schema, chunks_table=ct)),
                {"article_id": str(article_id)},
            )
            for i, content in enumerate(chunks):
                dense_val = _dense_vec_str(dense_vectors[i]) if dense_vectors else None
                sparse_val = (
                    _to_sparsevec_string(sparse_vectors[i], self._sparse_dim)
                    if sparse_vectors is not None and self._sparse_dim
                    else None
                )
                conn.execute(
                    text(_DML_INSERT_CHUNK.format(schema=schema, chunks_table=ct)),
                    {
                        "article_id": str(article_id),
                        "chunk_index": i,
                        "content": content,
                        "dense_vector": dense_val,
                        "sparse_vector": sparse_val,
                    },
                )
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Upsert failed for article {article_id}: {exc}") from exc
```

- [ ] **Step 3: Update imports**

```python
from chatbot_plugin_sdk.backends._pg_ddl import (
    _build_search_dense_sql,
    _build_search_sparse_sql,
    _build_upsert_article_sql,
    _DDL_CREATE_CHUNKS,
    _DDL_CREATE_ARTICLES,
    _DDL_CREATE_EXTENSION,
    _DDL_CREATE_SCHEMA,
    _DDL_IDX_SOURCE,
    _DDL_IDX_URL,
    _DDL_TABLE_EXISTS,
    _DML_DELETE_CHUNKS,
    _DML_INSERT_CHUNK,
    _split_article_fields,
    _check_dim_from_cols,
    _check_sparse_dim_from_cols,
    _dense_col_ddl,
    _dense_vec_str,
    _sparse_col_ddl,
    _to_sparsevec_string,
)
```

Remove: `_DML_UPSERT_ARTICLE`, `_DQL_SEARCH_DENSE`, `_DQL_SEARCH_SPARSE`.

Add `from typing import Any` at the top.

- [ ] **Step 4: Commit**

```bash
git add src/chatbot_plugin_sdk/backends/sync_pg.py
git commit -m "refactor: SyncPgBackend uses generic filters and article_columns"
```

---

### Task 5: Update `RetrieveProcessor`

**Files:**
- Modify: `src/chatbot_plugin_sdk/processors/retrieve.py`

- [ ] **Step 1: Update `retrieve()` signature**

Change `topic_id: str | None = None` to `filters: dict[str, Any] | None = None`:

```python
async def retrieve(
    self,
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
    min_rerank_score: float = 0.0,
    filters: dict[str, Any] | None = None,
) -> SearchResponse:
```

Add `from typing import Any` at the top of the file.

- [ ] **Step 2: Update all `search_dense` / `search_sparse` calls inside `retrieve()`**

Replace every `topic_id=topic_id` with `filters=filters`:

- Line 139: `self._backend.search_dense(dense_vecs[0], candidate_k, topic_id=topic_id)` → `self._backend.search_dense(dense_vecs[0], candidate_k, filters=filters)`
- Line 140: `self._backend.search_sparse(sparse_vecs[0], candidate_k, topic_id=topic_id)` → `self._backend.search_sparse(sparse_vecs[0], candidate_k, filters=filters)`
- Line 150: `self._backend.search_dense(dense_vecs[0], candidate_k, topic_id=topic_id)` → `self._backend.search_dense(dense_vecs[0], candidate_k, filters=filters)`
- Line 157: `self._backend.search_sparse(sparse_vecs[0], candidate_k, topic_id=topic_id)` → `self._backend.search_sparse(sparse_vecs[0], candidate_k, filters=filters)`

- [ ] **Step 3: Update log extra dict**

In the `logger.debug("retrieve_start", ...)` call, replace `"mode": ...` with the filters info:

```python
logger.debug(
    "retrieve_start",
    extra={
        "query_len": len(query),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "mode": "hybrid" if (self._dense and self._sparse) else "dense" if self._dense else "sparse",
        "reranker": type(self._reranker).__name__ if self._reranker else None,
        "filters": filters,
    },
)
```

- [ ] **Step 4: Update `ChunkResult` construction — remove `public_article_id` from SearchRow mapping**

The `SearchRow` no longer has `topic_id`, so the `ChunkResult` construction is already correct (it never set `topic_id` on `ChunkResult`). No change needed here.

- [ ] **Step 5: Run tests**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && python -m pytest src/tests/test_retrieve.py -v`
Expected: All tests pass (after fixing the test helper `_make_row` to not set `topic_id` and updating any `topic_id` references).

- [ ] **Step 6: Commit**

```bash
git add src/chatbot_plugin_sdk/processors/retrieve.py
git commit -m "refactor: RetrieveProcessor uses generic filters instead of topic_id"
```

---

### Task 6: Update `IngestProcessor`

**Files:**
- Modify: `src/chatbot_plugin_sdk/processors/ingest.py`

- [ ] **Step 1: Add `article_columns` parameter to `ingest()`**

Update the method signature:

```python
async def ingest(
    self,
    full_text: str,
    metadata: dict[str, Any] | None = None,
    article_columns: dict[str, Any] | None = None,
) -> None:
```

Update the docstring to document `article_columns`:

```
Args:
    full_text: Raw article text (HTML-stripped or plain).
    metadata:  Must contain ``url`` (str) for idempotent upsert keying.
               Keys that overlap with article columns are extracted automatically.
               Remaining keys are stored as JSONB metadata.
    article_columns: Explicit column values for the articles table, e.g.
                     ``{"topic_id": "uuid-xxx"}``. Keys must be known article
                     columns. Overrides any matching keys in metadata.
```

- [ ] **Step 2: Pass `article_columns` to `backend.upsert()`**

Change the upsert call at the bottom of `ingest()`:

```python
await self._backend.upsert(
    article_id, metadata, chunks, dense_vectors, sparse_vectors,
    article_columns=article_columns,
)
```

- [ ] **Step 3: Run ingest tests**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && python -m pytest src/tests/test_ingest.py -v`
Expected: Tests that call `backend.upsert` with positional args may need adjustment for the new `article_columns` keyword. The mock backend won't care about the extra kwarg, so most tests should pass.

- [ ] **Step 4: Commit**

```bash
git add src/chatbot_plugin_sdk/processors/ingest.py
git commit -m "refactor: IngestProcessor accepts article_columns for explicit column writes"
```

---

### Task 7: Update `DatabaseBackend.upsert` protocol signature

**Files:**
- Modify: `src/chatbot_plugin_sdk/backends/base.py`

- [ ] **Step 1: Add `article_columns` parameter to the upsert protocol**

```python
async def upsert(
    self,
    article_id: uuid.UUID,
    metadata: dict,
    chunks: list[str],
    dense_vectors: list[list[float]] | None,
    sparse_vectors: list[dict[str, float]] | None,
    article_columns: dict[str, Any] | None = None,
) -> None:
    """Insert-or-replace article + chunks inside a single transaction.

    Args:
        article_columns: Explicit column values for the articles table.
                          Keys must be in the known article column set.
                          These override matching keys from metadata.
    """
    ...
```

- [ ] **Step 2: Commit**

```bash
git add src/chatbot_plugin_sdk/backends/base.py
git commit -m "refactor: add article_columns to DatabaseBackend.upsert protocol"
```

---

### Task 8: Update contracts (requests)

**Files:**
- Modify: `src/chatbot_plugin_sdk/contracts/requests.py`

- [ ] **Step 1: Add `filters` field to `SearchRequest`**

```python
class SearchRequest(BaseModel):
    """POST /tools/search request."""
    query: str = Field(..., min_length=1, description="Raw query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top chunks to return")
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Column-level filters on the articles table, e.g. {\"topic_id\": \"uuid\"}",
    )
```

Add `from typing import Any` at the top of the file.

- [ ] **Step 2: Commit**

```bash
git add src/chatbot_plugin_sdk/contracts/requests.py
git commit -m "refactor: add filters field to SearchRequest contract"
```

---

### Task 9: Fix Article ORM model

**Files:**
- Modify: `src/chatbot_plugin_sdk/models/article.py`

- [ ] **Step 1: Add missing `public_article_id` and `topic_id` columns**

The ORM model is out of sync with the actual DDL schema. Add the missing columns:

```python
class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (
        Index("idx_articles_source", "source"),
        Index("idx_articles_url", "url"),
        {"schema": "vectors"},
    )

    id = Column(UUID(as_uuid=True), primary_key=True)
    url = Column(String, nullable=False, unique=True)
    title = Column(String, nullable=True)
    source = Column(String, nullable=True)
    public_article_id = Column(UUID(as_uuid=True), nullable=True)
    topic_id = Column(UUID(as_uuid=True), nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        server_onupdate=func.now(),
        default=lambda: datetime.now(timezone.utc),
    )
    chunks = relationship("ArticleChunk", back_populates="article", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title={self.title!r})>"
```

- [ ] **Step 2: Commit**

```bash
git add src/chatbot_plugin_sdk/models/article.py
git commit -m "fix: add missing public_article_id and topic_id to Article ORM model"
```

---

### Task 10: Update all tests

**Files:**
- Modify: `src/tests/test_retrieve.py`
- Modify: `src/tests/test_ingest.py`

- [ ] **Step 1: Update `_make_row` helper in `test_retrieve.py`**

Remove `topic_id` from `SearchRow` construction (it's no longer a field):

```python
def _make_row(chunk_id, article_id, idx, content, title, url, distance=0.2) -> SearchRow:
    return SearchRow(
        chunk_id=chunk_id, article_id=article_id, chunk_index=idx,
        content=content, title=title, url=url, distance=distance,
    )
```

This is already the current state — no change needed since `_make_row` doesn't set `topic_id`.

- [ ] **Step 2: Add tests for `filters` parameter in retrieve**

Add to `test_retrieve.py`:

```python
class TestRetrieveFilters:
    @pytest.mark.asyncio
    async def test_passes_filters_to_backend(self):
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = []
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            await retriever.retrieve("q", filters={"topic_id": "some-uuid"})

        call_kwargs = backend.search_dense.call_args.kwargs
        assert call_kwargs["filters"] == {"topic_id": "some-uuid"}

    @pytest.mark.asyncio
    async def test_filters_default_none(self):
        retriever, backend = _configured_retriever()
        backend.search_dense.return_value = []
        with patch.object(retriever._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            await retriever.retrieve("q")

        call_kwargs = backend.search_dense.call_args.kwargs
        assert call_kwargs.get("filters") is None

    @pytest.mark.asyncio
    async def test_hybrid_passes_filters_to_both_backends(self):
        backend = _mock_backend()
        backend.search_dense.return_value = [_make_row("c1", "a1", 0, "t", "T", "u", 0.1)]
        backend.search_sparse.return_value = [_make_row("c2", "a2", 0, "t", "T", "u", -0.9)]

        dense = EndpointProvider(url="http://x", dimension=768)
        sparse = MagicMock()
        sparse.dimension = 30522
        sparse.embed = AsyncMock(return_value=[{"0": 0.5, "1": 0.3}])

        retriever = RetrieveProcessor()
        retriever.configure(backend=backend, dense=dense, sparse=sparse)
        retriever._ready = True

        with patch.object(dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            await retriever.retrieve("q", filters={"source": "wiki"})

        dense_kwargs = backend.search_dense.call_args.kwargs
        sparse_kwargs = backend.search_sparse.call_args.kwargs
        assert dense_kwargs["filters"] == {"source": "wiki"}
        assert sparse_kwargs["filters"] == {"source": "wiki"}
```

- [ ] **Step 3: Add tests for `article_columns` in ingest**

Add to `test_ingest.py`:

```python
class TestIngestArticleColumns:
    @pytest.mark.asyncio
    async def test_passes_article_columns_to_backend(self):
        processor, backend = _configured_processor()
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                metadata={"url": "https://example.com/a", "title": "Test"},
                article_columns={"topic_id": "some-uuid"},
            )

        call_kwargs = backend.upsert.call_args.kwargs
        assert call_kwargs.get("article_columns") == {"topic_id": "some-uuid"}

    @pytest.mark.asyncio
    async def test_article_columns_default_none(self):
        processor, backend = _configured_processor()
        with patch.object(processor._dense, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
            await processor.ingest(
                "Hello world. " * 100,
                metadata={"url": "https://example.com/a", "title": "Test"},
            )

        call_kwargs = backend.upsert.call_args.kwargs
        assert call_kwargs.get("article_columns") is None
```

- [ ] **Step 4: Add tests for `_pg_ddl` helpers**

Add a new test file `src/tests/test_pg_ddl.py`:

```python
"""Tests for _pg_ddl helper functions."""
import pytest
from chatbot_plugin_sdk.backends._pg_ddl import (
    _ARTICLE_COLUMNS,
    _split_article_fields,
    _build_search_where,
    _build_upsert_article_sql,
    _build_search_dense_sql,
    _build_search_sparse_sql,
)
from chatbot_plugin_sdk.exceptions import DatabaseError


class TestArticleColumns:
    def test_known_columns(self):
        assert "topic_id" in _ARTICLE_COLUMNS
        assert "url" in _ARTICLE_COLUMNS
        assert "title" in _ARTICLE_COLUMNS
        assert "source" in _ARTICLE_COLUMNS
        assert "public_article_id" in _ARTICLE_COLUMNS


class TestSplitArticleFields:
    def test_splits_core_fields(self):
        metadata = {"url": "https://x.com", "title": "T", "extra_key": "val"}
        col_params, jsonb = _split_article_fields(metadata)
        assert col_params["url"] == "https://x.com"
        assert col_params["title"] == "T"
        assert jsonb == {"extra_key": "val"}

    def test_article_columns_override(self):
        metadata = {"url": "https://x.com"}
        col_params, jsonb = _split_article_fields(
            metadata, article_columns={"topic_id": "uuid-123"}
        )
        assert col_params["topic_id"] == "uuid-123"

    def test_article_columns_invalid_key_raises(self):
        with pytest.raises(DatabaseError, match="not a known article column"):
            _split_article_fields(
                {"url": "https://x.com"},
                article_columns={"nonexistent_col": "val"},
            )

    def test_no_jsonb_when_empty(self):
        metadata = {"url": "https://x.com", "title": "T"}
        _, jsonb = _split_article_fields(metadata)
        assert jsonb is None


class TestBuildSearchWhere:
    def test_no_filters(self):
        frag, params = _build_search_where(None)
        assert frag == ""
        assert params == {}

    def test_empty_filters(self):
        frag, params = _build_search_where({})
        assert frag == ""
        assert params == {}

    def test_single_filter(self):
        frag, params = _build_search_where({"source": "wiki"})
        assert "a.source = :_f_source" in frag
        assert params["_f_source"] == "wiki"

    def test_uuid_filter(self):
        frag, params = _build_search_where({"topic_id": "uuid-xxx"})
        assert "CAST(:_f_topic_id AS UUID)" in frag
        assert params["_f_topic_id"] == "uuid-xxx"

    def test_multiple_filters_are_anded(self):
        frag, params = _build_search_where({"source": "wiki", "topic_id": "uuid-xxx"})
        assert "AND" in frag
        assert "_f_source" in params
        assert "_f_topic_id" in params

    def test_invalid_filter_key_raises(self):
        with pytest.raises(DatabaseError, match="not a known article column"):
            _build_search_where({"nonexistent": "val"})


class TestBuildUpsertArticleSql:
    def test_basic_sql(self):
        col_params = {"url": "https://x.com", "title": "T"}
        sql = _build_upsert_article_sql("vectors", "articles", col_params)
        assert "INSERT INTO vectors.articles" in sql
        assert "ON CONFLICT (id) DO UPDATE SET" in sql
        assert ":url" in sql
        assert ":title" in sql

    def test_includes_topic_id_when_present(self):
        col_params = {"url": "https://x.com", "topic_id": "uuid-xxx"}
        sql = _build_upsert_article_sql("vectors", "articles", col_params)
        assert ":topic_id" in sql
        assert "CAST(:topic_id AS UUID)" in sql


class TestBuildSearchDenseSql:
    def test_no_filters(self):
        sql, params = _build_search_dense_sql("vectors", "articles", "chunks")
        assert "WHERE ac.dense_vector IS NOT NULL" in sql
        assert params == {}

    def test_with_filters(self):
        sql, params = _build_search_dense_sql(
            "vectors", "articles", "chunks",
            filters={"source": "wiki"},
        )
        assert "a.source = :_f_source" in sql
        assert params["_f_source"] == "wiki"
```

- [ ] **Step 5: Run all tests**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && python -m pytest src/tests/ -v`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/tests/test_retrieve.py src/tests/test_ingest.py src/tests/test_pg_ddl.py
git commit -m "test: update tests for generic filters and article_columns"
```

---

### Task 11: Final verification and cleanup

- [ ] **Step 1: Run full test suite**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && python -m pytest src/tests/ -v --tb=short`
Expected: All tests pass with no warnings.

- [ ] **Step 2: Verify no `topic_id` references remain in protocol/processor code**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && grep -rn "topic_id" src/chatbot_plugin_sdk/backends/base.py src/chatbot_plugin_sdk/processors/ src/chatbot_plugin_sdk/contracts/`
Expected: No matches (topic_id only exists in DDL and the ORM model, which is correct — the column exists, it's just no longer hardcoded in queries).

- [ ] **Step 3: Verify `topic_id` still exists in DDL (it's a valid column)**

Run: `cd /home/pegaai/Desktop/sdd/chatbot-plugin-sdk && grep -n "topic_id" src/chatbot_plugin_sdk/backends/_pg_ddl.py`
Expected: `topic_id` appears only in `_ARTICLE_COLUMNS`, `_DDL_CREATE_ARTICLES`, and `uuid_cols` inside helper functions. The old hardcoded `_DML_UPSERT_ARTICLE` and `_DQL_SEARCH_*` constants are gone.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup for generic filter mechanism refactor"
```

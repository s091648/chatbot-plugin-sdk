"""Tests for _pg_ddl helper functions."""
import pytest

from chatbot_plugin_sdk.backends._pg_ddl import (
    _ARTICLE_COLUMNS,
    _prepare_upsert_params,
    _extract_article_metadata,
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


class TestPrepareUpsertParams:
    def test_metadata_not_promoted_to_sql_cols(self):
        """metadata keys are never auto-promoted to SQL columns."""
        metadata = {"url": "https://x.com", "title": "T", "extra_key": "val"}
        col_params, jsonb = _prepare_upsert_params(metadata)
        assert col_params == {}
        assert jsonb == {"url": "https://x.com", "title": "T", "extra_key": "val"}

    def test_article_columns_provide_sql_values(self):
        metadata = {"url": "https://x.com"}
        col_params, jsonb = _prepare_upsert_params(
            metadata, article_columns={"topic_id": "uuid-123"}
        )
        assert col_params["topic_id"] == "uuid-123"
        assert jsonb == {"url": "https://x.com"}

    def test_article_columns_invalid_key_raises(self):
        with pytest.raises(DatabaseError, match="not a known article column"):
            _prepare_upsert_params(
                {"url": "https://x.com"},
                article_columns={"nonexistent_col": "val"},
            )

    def test_no_jsonb_when_empty(self):
        col_params, jsonb = _prepare_upsert_params()
        assert col_params == {}
        assert jsonb is None

    def test_metadata_none(self):
        col_params, jsonb = _prepare_upsert_params(
            metadata=None, article_columns={"url": "https://x.com"}
        )
        assert col_params == {"url": "https://x.com"}
        assert jsonb is None


class TestExtractArticleMetadata:
    def test_extracts_article_columns(self):
        row = {
            "chunk_id": "c1",
            "article_id": "a1",
            "chunk_index": 0,
            "content": "text",
            "distance": 0.2,
            "title": "My Article",
            "url": "https://example.com",
            "source": "wiki",
            "public_article_id": "uuid-123",
            "topic_id": None,
        }
        result = _extract_article_metadata(row, _ARTICLE_COLUMNS)
        assert result == {
            "title": "My Article",
            "url": "https://example.com",
            "source": "wiki",
            "public_article_id": "uuid-123",
        }
        assert "topic_id" not in result  # None values excluded
        assert "chunk_id" not in result
        assert "content" not in result

    def test_empty_when_no_article_columns_present(self):
        row = {"chunk_id": "c1", "article_id": "a1", "content": "text", "distance": 0.2}
        result = _extract_article_metadata(row, _ARTICLE_COLUMNS)
        assert result == {}


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

    def test_includes_all_article_columns_in_select(self):
        sql, _ = _build_search_dense_sql("vectors", "articles", "chunks")
        for col in _ARTICLE_COLUMNS:
            assert f"a.{col}" in sql

    def test_with_filters(self):
        sql, params = _build_search_dense_sql(
            "vectors", "articles", "chunks",
            filters={"source": "wiki"},
        )
        assert "a.source = :_f_source" in sql
        assert params["_f_source"] == "wiki"


class TestBuildSearchSparseSql:
    def test_no_filters(self):
        sql, params = _build_search_sparse_sql("vectors", "articles", "chunks")
        assert "WHERE ac.sparse_vector IS NOT NULL" in sql
        assert "sparse_vector <#>" in sql
        assert params == {}

    def test_includes_all_article_columns_in_select(self):
        sql, _ = _build_search_sparse_sql("vectors", "articles", "chunks")
        for col in _ARTICLE_COLUMNS:
            assert f"a.{col}" in sql

    def test_with_filters(self):
        sql, params = _build_search_sparse_sql(
            "vectors", "articles", "chunks",
            filters={"topic_id": "uuid-xxx"},
        )
        assert "CAST(:_f_topic_id AS UUID)" in sql
        assert params["_f_topic_id"] == "uuid-xxx"

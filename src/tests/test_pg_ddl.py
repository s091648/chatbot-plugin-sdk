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


class TestBuildSearchSparseSql:
    def test_no_filters(self):
        sql, params = _build_search_sparse_sql("vectors", "articles", "chunks")
        assert "WHERE ac.sparse_vector IS NOT NULL" in sql
        assert "sparse_vector <#>" in sql
        assert params == {}

    def test_with_filters(self):
        sql, params = _build_search_sparse_sql(
            "vectors", "articles", "chunks",
            filters={"topic_id": "uuid-xxx"},
        )
        assert "CAST(:_f_topic_id AS UUID)" in sql
        assert params["_f_topic_id"] == "uuid-xxx"

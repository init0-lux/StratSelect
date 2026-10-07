from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_sql_layer_contains_table_indexes_and_adaptive_function():
    schema = (ROOT / "db/schema.sql").read_text()
    indexes = (ROOT / "db/indexes.sql").read_text()
    adaptive = (ROOT / "db/adaptive_search.sql").read_text()

    assert "embedding vector(64)" in schema
    assert "CREATE INDEX" in indexes
    assert "USING hnsw" in indexes
    assert "CREATE OR REPLACE FUNCTION adaptive_search" in adaptive
    assert "ef_search" in adaptive
    assert "RETURN QUERY" in adaptive


def test_readme_documents_docker_run_and_calibrated_latency():
    readme = (ROOT / "README.md").read_text()
    assert "docker compose run --rm benchmark" in readme
    assert "calibrated reference" in readme
    assert "Euclidean" in readme

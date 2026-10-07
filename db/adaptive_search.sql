CREATE OR REPLACE FUNCTION adaptive_search(
    query_embedding vector(64),
    requested_category text,
    max_price numeric,
    result_k integer DEFAULT 10
)
RETURNS TABLE (id bigint, name text, category text, price numeric, embedding vector(64))
LANGUAGE plpgsql
AS $$
DECLARE
    total_rows bigint;
    matching_rows bigint;
    selectivity numeric;
    exact_cost numeric;
    ef_search integer;
    ann_cost numeric;
BEGIN
    SELECT reltuples::bigint INTO total_rows FROM pg_class WHERE oid = 'products'::regclass;
    SELECT count(*) INTO matching_rows FROM products WHERE products.category = requested_category AND products.price < max_price;
    total_rows := greatest(total_rows, 1);
    selectivity := matching_rows::numeric / total_rows;
    exact_cost := total_rows * selectivity;
    ef_search := ceil(3 * result_k / greatest(selectivity, 0.000001));
    ann_cost := 5 * ef_search;

    IF exact_cost <= ann_cost THEN
        RETURN QUERY
        SELECT p.id, p.name, p.category, p.price, p.embedding
        FROM products p
        WHERE p.category = requested_category AND p.price < max_price
        ORDER BY p.embedding <-> query_embedding, p.id
        LIMIT result_k;
    ELSE
        PERFORM set_config('hnsw.ef_search', ef_search::text, true);
        RETURN QUERY
        SELECT p.id, p.name, p.category, p.price, p.embedding
        FROM products p
        WHERE p.category = requested_category AND p.price < max_price
        ORDER BY p.embedding <-> query_embedding, p.id
        LIMIT result_k;
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS products_category_idx ON products (category);
CREATE INDEX IF NOT EXISTS products_price_idx ON products (price);
CREATE INDEX IF NOT EXISTS products_embedding_hnsw_idx
    ON products USING hnsw (embedding vector_l2_ops);

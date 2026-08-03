import duckdb
import pandas as pd

print("Loading dataset...")

con = duckdb.connect("strategiz.db")

datasets = {
    "orders": "data/olist_orders_dataset.csv",
    "customers": "data/olist_customers_dataset.csv",
    "order_items": "data/olist_order_items_dataset.csv",
    "payments": "data/olist_order_payments_dataset.csv",
    "products": "data/olist_products_dataset.csv",
    "sellers": "data/olist_sellers_dataset.csv"
}

for table, path in datasets.items():
    print(f"Loading {table}...")
    df = pd.read_csv(path)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")

print("✅ Database created successfully!")
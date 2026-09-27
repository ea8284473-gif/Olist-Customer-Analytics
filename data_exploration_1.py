import pandas as pd

customers = pd.read_csv("data/olist_customers_dataset.csv")

print(customers.head())

print("Number of rows and columns:")
print(customers.shape)

print("\nColumn names:")
print(customers.columns.tolist())

print("\nMissing values:")
print(customers.isnull().sum())

print("\nDuplicate rows:")
print(customers.duplicated().sum())

print("\nUnique customers:")
print(customers["customer_unique_id"].nunique())

print("\nCustomers by state:")
print(customers["customer_state"].value_counts().head(10))

# Load orders dataset
orders = pd.read_csv("data/olist_orders_dataset.csv")

print("\n--- ORDERS DATASET ---")

print("\nRows and columns:")
print(orders.shape)

print("\nColumn names:")
print(orders.columns.tolist())

print("\nMissing values:")
print(orders.isnull().sum())

print("\nDuplicate rows:")
print(orders.duplicated().sum())

print("\nOrder status:")
print(orders["order_status"].value_counts())

# Convert delivery dates to datetime
orders["order_delivered_customer_date"] = pd.to_datetime(
    orders["order_delivered_customer_date"]
)

orders["order_estimated_delivery_date"] = pd.to_datetime(
    orders["order_estimated_delivery_date"]
)

# Select delivered orders with actual delivery dates
delivered = orders[
    (orders["order_status"] == "delivered") &
    (orders["order_delivered_customer_date"].notna())
].copy()

# Identify late deliveries
delivered["is_late"] = (
    delivered["order_delivered_customer_date"] >
    delivered["order_estimated_delivery_date"]
)

print("\n--- DELIVERY ANALYSIS ---")

print("Delivered orders:", len(delivered))

print("Late deliveries:", delivered["is_late"].sum())

print(
    "Late delivery rate:",
    round(delivered["is_late"].mean() * 100, 2),
    "%"
)

# Load reviews dataset
reviews = pd.read_csv(
    "data/olist_order_reviews_dataset.csv"
)

# Keep one review score per order
reviews_per_order = (
    reviews.groupby("order_id")["review_score"]
    .mean()
    .reset_index()
)

# Merge delivered orders with reviews
delivery_reviews = delivered.merge(
    reviews_per_order,
    on="order_id",
    how="inner"
)

print("\n--- DELIVERY VS CUSTOMER SATISFACTION ---")

print("\nAverage review score:")
print(
    delivery_reviews.groupby("is_late")["review_score"]
    .mean()
)

print("\nNumber of orders:")
print(
    delivery_reviews.groupby("is_late")["order_id"]
    .nunique()
)

# Merge delivery data with customer locations
delivery_location = delivered.merge(
    customers[
        ["customer_id", "customer_state"]
    ],
    on="customer_id",
    how="left"
)

# Calculate delivery performance by state
state_analysis = (
    delivery_location
    .groupby("customer_state")
    .agg(
        total_orders=("order_id", "nunique"),
        late_orders=("is_late", "sum"),
        late_rate=("is_late", "mean")
    )
)

# Convert late rate to percentage
state_analysis["late_rate"] = (
    state_analysis["late_rate"] * 100
).round(2)

# Show states with at least 100 orders
state_analysis = state_analysis[
    state_analysis["total_orders"] >= 100
]

# Sort by highest late delivery rate
state_analysis = state_analysis.sort_values(
    "late_rate",
    ascending=False
)

print("\n--- DELIVERY PERFORMANCE BY STATE ---")
print(state_analysis.head(10))

# Convert purchase date to datetime
delivery_location["order_purchase_timestamp"] = (
    pd.to_datetime(
        delivery_location["order_purchase_timestamp"]
    )
)

# Calculate delivery duration in days
delivery_location["delivery_days"] = (
    delivery_location["order_delivered_customer_date"]
    - delivery_location["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# Calculate average delivery duration by state
delivery_time_by_state = (
    delivery_location
    .groupby("customer_state")
    .agg(
        total_orders=("order_id", "nunique"),
        avg_delivery_days=("delivery_days", "mean"),
        late_rate=("is_late", "mean")
    )
)

delivery_time_by_state["late_rate"] *= 100

delivery_time_by_state = (
    delivery_time_by_state[
        delivery_time_by_state["total_orders"] >= 100
    ]
    .sort_values(
        "avg_delivery_days",
        ascending=False
    )
)

print("\n--- DELIVERY TIME BY STATE ---")

print(delivery_time_by_state.head(10).round(2))

import matplotlib.pyplot as plt

# Select top 10 states by late delivery rate
top_states = state_analysis.head(10)

# Create chart
plt.figure(figsize=(12, 6))

plt.bar(
    top_states.index,
    top_states["late_rate"],
    color="steelblue"
)

plt.title("Late Delivery Rate by State")
plt.xlabel("Customer State")
plt.ylabel("Late Delivery Rate (%)")

plt.axhline(
    y=8.11,
    color="red",
    linestyle="--",
    label="Overall Late Rate: 8.11%"
)

plt.legend()
plt.tight_layout()

# Save chart
plt.savefig(
    "late_delivery_by_state.png",
    dpi=300
)

plt.show()


# Customer satisfaction chart

review_comparison = (
    delivery_reviews.groupby("is_late")["review_score"]
    .mean()
)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    ["On-time Delivery", "Late Delivery"],
    [
        review_comparison[False],
        review_comparison[True]
    ],
    color=["#2E86AB", "#E76F51"]
)

plt.title("Impact of Delivery Delays on Customer Ratings")
plt.ylabel("Average Review Score")
plt.ylim(0, 5)

# Display values above bars
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{bar.get_height():.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "delivery_vs_satisfaction.png",
    dpi=300
)

plt.show()


# Load order items dataset
order_items = pd.read_csv(
    "data/olist_order_items_dataset.csv"
)

print("\n--- ORDER ITEMS DATASET ---")

print("\nRows and columns:")
print(order_items.shape)

print("\nColumn names:")
print(order_items.columns.tolist())

print("\nMissing values:")
print(order_items.isnull().sum())

print("\nDuplicate rows:")
print(order_items.duplicated().sum())

print("\nFirst 5 rows:")
print(order_items.head())

print("\nUnique sellers:")
print(order_items["seller_id"].nunique())

print("\nUnique products:")
print(order_items["product_id"].nunique())

print("\nUnique orders:")
print(order_items["order_id"].nunique())


# ====================================
# SELLER DELIVERY PERFORMANCE
# ====================================

# Merge orders with order items
seller_orders = order_items.merge(
    orders[
        [
            "order_id",
            "order_status",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    ],
    on="order_id",
    how="left"
)

# Convert dates
seller_orders["order_delivered_customer_date"] = pd.to_datetime(
    seller_orders["order_delivered_customer_date"]
)

seller_orders["order_estimated_delivery_date"] = pd.to_datetime(
    seller_orders["order_estimated_delivery_date"]
)

# Keep delivered orders only
seller_orders = seller_orders[
    seller_orders["order_status"] == "delivered"
].copy()

# Calculate late delivery
seller_orders["is_late"] = (
    seller_orders["order_delivered_customer_date"]
    > seller_orders["order_estimated_delivery_date"]
)

# Count each order once per seller
seller_orders = seller_orders.drop_duplicates(
    subset=["seller_id", "order_id"]
)

# Seller performance
seller_performance = seller_orders.groupby(
    "seller_id"
).agg(
    total_orders=("order_id", "count"),
    late_orders=("is_late", "sum"),
    late_rate=("is_late", "mean")
)

seller_performance["late_rate"] *= 100

# Filter sellers with at least 30 orders
seller_performance = seller_performance[
    seller_performance["total_orders"] >= 30
]

# Sort by late delivery rate
seller_performance = seller_performance.sort_values(
    "late_rate",
    ascending=False
)

print("\n--- SELLER DELIVERY PERFORMANCE ---")

print("\nSellers with at least 30 orders:")
print(len(seller_performance))

print("\nTop 10 sellers by late delivery rate:")
print(seller_performance.head(10).round(2))


# ====================================
# FREIGHT COST VS DELIVERY DELAYS
# ====================================

# Calculate total freight per order
order_freight = order_items.groupby(
    "order_id",
    as_index=False
)["freight_value"].sum()

# Merge with orders
freight_analysis = orders.merge(
    order_freight,
    on="order_id",
    how="inner"
)

# Keep delivered orders
freight_analysis = freight_analysis[
    freight_analysis["order_status"] == "delivered"
].copy()

# Convert dates
freight_analysis["order_delivered_customer_date"] = pd.to_datetime(
    freight_analysis["order_delivered_customer_date"]
)

freight_analysis["order_estimated_delivery_date"] = pd.to_datetime(
    freight_analysis["order_estimated_delivery_date"]
)

# Calculate delays
freight_analysis["is_late"] = (
    freight_analysis["order_delivered_customer_date"]
    > freight_analysis["order_estimated_delivery_date"]
)

# Divide freight cost into 4 groups
freight_analysis["freight_group"] = pd.qcut(
    freight_analysis["freight_value"],
    q=4,
    duplicates="drop"
)

# Analyze each group
freight_performance = freight_analysis.groupby(
    "freight_group",
    observed=True
).agg(
    total_orders=("order_id", "count"),
    average_freight=("freight_value", "mean"),
    late_orders=("is_late", "sum"),
    late_rate=("is_late", "mean")
)

freight_performance["late_rate"] *= 100

print("\n--- FREIGHT COST VS DELIVERY ---")
print(freight_performance.round(2))


# ====================================
# GEOLOCATION DATA EXPLORATION
# ====================================

geo = pd.read_csv(
    "data/olist_geolocation_dataset.csv"
)

sellers = pd.read_csv(
    "data/olist_sellers_dataset.csv"
)

print("\n--- GEOLOCATION DATASET ---")

print("\nRows and columns:")
print(geo.shape)

print("\nColumn names:")
print(geo.columns.tolist())

print("\nMissing values:")
print(geo.isnull().sum())

print("\nDuplicate rows:")
print(geo.duplicated().sum())

print("\nUnique ZIP codes:")
print(
    geo["geolocation_zip_code_prefix"].nunique()
)

print("\n--- SELLERS DATASET ---")

print("\nRows and columns:")
print(sellers.shape)

print("\nColumn names:")
print(sellers.columns.tolist())

print("\nMissing values:")
print(sellers.isnull().sum())

print("\nUnique sellers:")
print(sellers["seller_id"].nunique())


import numpy as np

# ====================================
# DISTANCE ANALYSIS
# ====================================

# Representative coordinates per ZIP code
geo_clean = geo.groupby(
    "geolocation_zip_code_prefix",
    as_index=False
).agg(
    lat=("geolocation_lat", "median"),
    lng=("geolocation_lng", "median")
)

# Attach seller ZIP codes
distance_data = order_items.merge(
    sellers[["seller_id", "seller_zip_code_prefix"]],
    on="seller_id",
    how="left"
)

# Attach customer ZIP codes
distance_data = distance_data.merge(
    orders[["order_id", "customer_id", "order_status",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"]],
    on="order_id",
    how="left"
)

distance_data = distance_data.merge(
    customers[["customer_id", "customer_zip_code_prefix"]],
    on="customer_id",
    how="left"
)

# Seller coordinates
distance_data = distance_data.merge(
    geo_clean.rename(columns={
        "geolocation_zip_code_prefix": "seller_zip_code_prefix",
        "lat": "seller_lat",
        "lng": "seller_lng"
    }),
    on="seller_zip_code_prefix",
    how="left"
)

# Customer coordinates
distance_data = distance_data.merge(
    geo_clean.rename(columns={
        "geolocation_zip_code_prefix": "customer_zip_code_prefix",
        "lat": "customer_lat",
        "lng": "customer_lng"
    }),
    on="customer_zip_code_prefix",
    how="left"
)

# Keep delivered orders with coordinates
distance_data = distance_data[
    distance_data["order_status"] == "delivered"
].dropna(subset=[
    "seller_lat", "seller_lng",
    "customer_lat", "customer_lng",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]).copy()

# Haversine distance
lat1 = np.radians(distance_data["seller_lat"])
lon1 = np.radians(distance_data["seller_lng"])
lat2 = np.radians(distance_data["customer_lat"])
lon2 = np.radians(distance_data["customer_lng"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2)
    * np.sin(dlon / 2) ** 2
)

a = np.clip(a, 0, 1)

distance_data["distance_km"] = (
    6371 * 2 * np.arcsin(np.sqrt(a))
)

# Late delivery
distance_data["is_late"] = (
    pd.to_datetime(distance_data["order_delivered_customer_date"])
    > pd.to_datetime(distance_data["order_estimated_delivery_date"])
)

# Distance groups
distance_data["distance_group"] = pd.cut(
    distance_data["distance_km"],
    bins=[0, 100, 300, 700, 1500, np.inf],
    labels=[
        "0-100 km",
        "100-300 km",
        "300-700 km",
        "700-1500 km",
        "1500+ km"
    ],
    include_lowest=True
)

# Analyze distance
distance_performance = distance_data.groupby(
    "distance_group",
    observed=True
).agg(
    total_items=("order_id", "count"),
    average_distance=("distance_km", "mean"),
    average_freight=("freight_value", "mean"),
    late_rate=("is_late", "mean")
)

distance_performance["late_rate"] *= 100

print("\n--- DISTANCE ANALYSIS ---")
print(distance_performance.round(2))


# ====================================
# DELIVERY TIME VS DISTANCE
# ====================================

# Get purchase dates
distance_data = distance_data.merge(
    orders[[
        "order_id",
        "order_purchase_timestamp"
    ]],
    on="order_id",
    how="left"
)

# Convert purchase dates
distance_data["order_purchase_timestamp"] = pd.to_datetime(
    distance_data["order_purchase_timestamp"]
)

# Convert delivery dates
distance_data["order_delivered_customer_date"] = pd.to_datetime(
    distance_data["order_delivered_customer_date"]
)

distance_data["order_estimated_delivery_date"] = pd.to_datetime(
    distance_data["order_estimated_delivery_date"]
)

# Actual delivery time
distance_data["delivery_days"] = (
    distance_data["order_delivered_customer_date"]
    - distance_data["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# Estimated delivery time
distance_data["estimated_days"] = (
    distance_data["order_estimated_delivery_date"]
    - distance_data["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# Remove invalid durations
distance_data = distance_data[
    (distance_data["delivery_days"] >= 0)
    & (distance_data["estimated_days"] >= 0)
].copy()

# Delivery performance by distance
delivery_time_analysis = distance_data.groupby(
    "distance_group",
    observed=True
).agg(
    total_items=("order_id", "count"),
    avg_actual_days=("delivery_days", "mean"),
    avg_estimated_days=("estimated_days", "mean"),
    late_rate=("is_late", "mean")
)

delivery_time_analysis["late_rate"] *= 100

print("\n--- DELIVERY TIME VS DISTANCE ---")

print(delivery_time_analysis.round(2))


# =====================================
# DELIVERY ESTIMATION ACCURACY
# =====================================

distance_data["delivery_gap_days"] = (
    distance_data["order_delivered_customer_date"]
    - distance_data["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

accuracy_analysis = distance_data.groupby(
    "distance_group",
    observed=True
).agg(
    total_items=("order_id", "count"),
    avg_delivery_gap=("delivery_gap_days", "mean"),
    median_delivery_gap=("delivery_gap_days", "median"),
    late_rate=("is_late", "mean")
)

accuracy_analysis["late_rate"] *= 100

print("\n--- DELIVERY ESTIMATION ACCURACY ---")
print(accuracy_analysis.round(2))
print(accuracy_analysis.to_string())


# =====================================
# ORDER-LEVEL DISTANCE ANALYSIS
# =====================================

# Keep the maximum seller-customer
# distance for each order

order_distance = (
    distance_data
    .groupby("order_id", as_index=False)
    .agg(
        max_distance_km=("distance_km", "max")
    )
)

# Merge with delivered orders
order_level = delivered.merge(
    order_distance,
    on="order_id",
    how="inner"
)

# Create distance groups
order_level["distance_group"] = pd.cut(
    order_level["max_distance_km"],
    bins=[0, 100, 300, 700, 1500, np.inf],
    labels=[
        "0-100 km",
        "100-300 km",
        "300-700 km",
        "700-1500 km",
        "1500+ km"
    ],
    include_lowest=True
)

# Calculate late delivery rate
order_distance_analysis = (
    order_level
    .groupby("distance_group", observed=True)
    .agg(
        total_orders=("order_id", "nunique"),
        late_orders=("is_late", "sum"),
        late_rate=("is_late", "mean")
    )
)

order_distance_analysis["late_rate"] *= 100

print("\n--- ORDER LEVEL DISTANCE ANALYSIS ---")

print(order_distance_analysis.round(2).to_string())


# =====================================
# DISTANCE VS CUSTOMER SATISFACTION
# =====================================

# Merge order-level distance with reviews
distance_reviews = order_level.merge(
    reviews_per_order,
    on="order_id",
    how="inner"
)

# Analyze ratings by distance and delay
distance_satisfaction = (
    distance_reviews
    .groupby(
        ["distance_group", "is_late"],
        observed=True
    )
    .agg(
        total_orders=("order_id", "nunique"),
        avg_review_score=("review_score", "mean")
    )
    .reset_index()
)

print("\n--- DISTANCE VS CUSTOMER SATISFACTION ---")

print(
    distance_satisfaction
    .round(2)
    .to_string(index=False)
)


# =====================================
# SELLER VS CARRIER DELIVERY ANALYSIS
# =====================================

print("\n--- SELLER VS CARRIER ANALYSIS ---")

# Create a separate dataset
delivery_stages = delivered.copy()

# Convert dates
date_columns = [
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date"
]

for col in date_columns:
    delivery_stages[col] = pd.to_datetime(
        delivery_stages[col]
    )

# Calculate processing time
delivery_stages["processing_days"] = (
    delivery_stages["order_delivered_carrier_date"]
    - delivery_stages["order_approved_at"]
).dt.total_seconds() / 86400

# Calculate carrier delivery time
delivery_stages["carrier_days"] = (
    delivery_stages["order_delivered_customer_date"]
    - delivery_stages["order_delivered_carrier_date"]
).dt.total_seconds() / 86400

# Remove missing and invalid durations
delivery_stages = delivery_stages.dropna(
    subset=["processing_days", "carrier_days"]
)

delivery_stages = delivery_stages[
    (delivery_stages["processing_days"] >= 0)
    & (delivery_stages["carrier_days"] >= 0)
].copy()

# Compare late and on-time orders
stage_analysis = delivery_stages.groupby(
    "is_late"
).agg(
    total_orders=("order_id", "nunique"),
    avg_processing_days=("processing_days", "mean"),
    avg_carrier_days=("carrier_days", "mean"),
    median_processing_days=("processing_days", "median"),
    median_carrier_days=("carrier_days", "median")
)

print("\n--- PROCESSING VS CARRIER TIME ---")
print(stage_analysis.round(2))

# Overall averages
print("\nOverall average processing time:")
print(
    round(delivery_stages["processing_days"].mean(), 2)
)

print("\nOverall average carrier time:")
print(
    round(delivery_stages["carrier_days"].mean(), 2)
)


# =====================================
# VISUALIZATION
# =====================================

stage_chart = stage_analysis[
    ["avg_processing_days", "avg_carrier_days"]
].copy()

stage_chart.index = [
    "On-time",
    "Late"
]

stage_chart.columns = [
    "Processing Time",
    "Carrier Time"
]

ax = stage_chart.plot(
    kind="bar",
    figsize=(10, 6),
    color=["#2E86AB", "#E76F51"]
)

plt.title("Processing vs Carrier Delivery Time")
plt.xlabel("Delivery Status")
plt.ylabel("Average Days")
plt.xticks(rotation=0)

# Display values
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3
    )

plt.legend()
plt.tight_layout()

plt.savefig(
    "processing_vs_carrier.png",
    dpi=300
)

plt.show()


# =====================================
# SELLER SHIPPING DEADLINE ANALYSIS
# =====================================

print("\n--- SELLER SHIPPING DEADLINE ---")

# Convert shipping deadline to datetime
order_items["shipping_limit_date"] = pd.to_datetime(
    order_items["shipping_limit_date"]
)

# Get one shipping deadline per order
shipping_deadlines = (
    order_items.groupby("order_id", as_index=False)
    ["shipping_limit_date"].max()
)

# Merge with delivered orders
shipping_analysis = delivered.merge(
    shipping_deadlines,
    on="order_id",
    how="inner"
)

# Convert carrier date
shipping_analysis["order_delivered_carrier_date"] = (
    pd.to_datetime(
        shipping_analysis["order_delivered_carrier_date"]
    )
)

# Remove missing dates
shipping_analysis = shipping_analysis.dropna(
    subset=[
        "shipping_limit_date",
        "order_delivered_carrier_date"
    ]
).copy()

# Identify late handovers
shipping_analysis["seller_late"] = (
    shipping_analysis["order_delivered_carrier_date"]
    > shipping_analysis["shipping_limit_date"]
)

# Seller late handover rate
print("\nTotal orders:")
print(len(shipping_analysis))

print("\nOrders handed over after deadline:")
print(shipping_analysis["seller_late"].sum())

print("\nLate handover rate:")
print(
    round(
        shipping_analysis["seller_late"].mean() * 100,
        2
    )
)

# Compare seller handover with final delivery
seller_impact = shipping_analysis.groupby(
    "seller_late"
).agg(
    total_orders=("order_id", "nunique"),
    late_deliveries=("is_late", "sum"),
    late_delivery_rate=("is_late", "mean")
)

seller_impact["late_delivery_rate"] *= 100

print("\n--- SELLER HANDOVER VS FINAL DELIVERY ---")
print(seller_impact.round(2))


# =====================================
# DELIVERY DELAY CLASSIFICATION
# =====================================

import numpy as np

print("\n--- DELIVERY DELAY CLASSIFICATION ---")

conditions = [
    (~shipping_analysis["seller_late"]) &
    (~shipping_analysis["is_late"]),

    (shipping_analysis["seller_late"]) &
    (~shipping_analysis["is_late"]),

    (~shipping_analysis["seller_late"]) &
    (shipping_analysis["is_late"]),

    (shipping_analysis["seller_late"]) &
    (shipping_analysis["is_late"])
]

labels = [
    "A - Both On Time",
    "B - Late Handover Only",
    "C - Late Delivery Only",
    "D - Both Late"
]

shipping_analysis["delay_category"] = np.select(
    conditions,
    labels,
    default="Unknown"
)

category_analysis = (
    shipping_analysis["delay_category"]
    .value_counts()
    .reindex(labels, fill_value=0)
)

category_percentage = (
    category_analysis / len(shipping_analysis) * 100
)

results = pd.DataFrame({
    "total_orders": category_analysis,
    "percentage": category_percentage.round(2)
})

print(results)

# Visualization
plt.figure(figsize=(11, 6))

bars = plt.bar(
    results.index,
    results["total_orders"],
    color=["#2A9D8F", "#E9C46A",
           "#E76F51", "#C44536"]
)

plt.title("Delivery Delay Classification")
plt.ylabel("Number of Orders")
plt.xticks(rotation=15)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 300,
        f"{int(bar.get_height()):,}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "delivery_delay_classification.png",
    dpi=300
)

plt.show()


# =====================================
# SELLER HANDOVER PERFORMANCE
# =====================================

print("\n--- SELLER HANDOVER PERFORMANCE ---")

# Prepare order items with carrier dates
seller_handover = order_items.merge(
    delivered[
        ["order_id", "order_delivered_carrier_date"]
    ],
    on="order_id",
    how="inner"
)

# Convert dates
seller_handover["shipping_limit_date"] = pd.to_datetime(
    seller_handover["shipping_limit_date"]
)

seller_handover["order_delivered_carrier_date"] = pd.to_datetime(
    seller_handover["order_delivered_carrier_date"]
)

# Remove missing dates
seller_handover = seller_handover.dropna(
    subset=[
        "shipping_limit_date",
        "order_delivered_carrier_date"
    ]
).copy()

# Calculate late handover
seller_handover["seller_late"] = (
    seller_handover["order_delivered_carrier_date"]
    > seller_handover["shipping_limit_date"]
)

# Count each order once per seller
seller_handover = seller_handover.drop_duplicates(
    subset=["seller_id", "order_id"]
)

# Calculate seller performance
seller_handover_performance = (
    seller_handover.groupby("seller_id")
    .agg(
        total_orders=("order_id", "nunique"),
        late_handovers=("seller_late", "sum"),
        late_handover_rate=("seller_late", "mean")
    )
)

seller_handover_performance["late_handover_rate"] *= 100

# Filter sellers with at least 30 orders
active_sellers = seller_handover_performance[
    seller_handover_performance["total_orders"] >= 30
].copy()

print("\nSellers with at least 30 orders:")
print(len(active_sellers))

# Top sellers by late handover rate
print("\n--- HIGHEST LATE HANDOVER RATES ---")

print(
    active_sellers.sort_values(
        "late_handover_rate",
        ascending=False
    ).head(10).round(2)
)


# =====================================
# SELLERS WITH MOST LATE HANDOVERS
# =====================================

print("\n--- SELLERS WITH MOST LATE HANDOVERS ---")

top_late_sellers = (
    active_sellers.sort_values(
        "late_handovers",
        ascending=False
    ).head(10)
)

print(top_late_sellers.round(2))

# Calculate concentration
total_late_handovers = (
    seller_handover["seller_late"].sum()
)

top10_late_handovers = (
    top_late_sellers["late_handovers"].sum()
)

contribution = (
    top10_late_handovers
    / total_late_handovers
    * 100
)

print("\nTotal late handovers:")
print(total_late_handovers)

print("\nLate handovers from top 10 sellers:")
print(top10_late_handovers)

print("\nTop 10 contribution (%):")
print(round(contribution, 2))


# =====================================
# VISUALIZATION
# =====================================

plt.figure(figsize=(12, 6))

top_sellers_chart = (
    top_late_sellers.sort_values(
        "late_handovers",
        ascending=True
    )
)

plt.barh(
    top_sellers_chart.index,
    top_sellers_chart["late_handovers"],
    color="#E76F51"
)

plt.title("Top 10 Sellers by Late Handovers")
plt.xlabel("Number of Late Handovers")
plt.ylabel("Seller ID")

plt.tight_layout()

plt.savefig(
    "top_sellers_late_handovers.png",
    dpi=300
)

plt.show()


# =====================================
# SELLER GEOGRAPHIC PERFORMANCE
# =====================================

# Merge seller performance with locations
seller_geo = seller_handover_performance.merge(
    sellers[
        ["seller_id", "seller_state"]
    ],
    on="seller_id",
    how="left"
)

# Analyze seller performance by state
seller_state_analysis = (
    seller_geo.groupby("seller_state")
    .agg(
        total_sellers=("seller_id", "nunique"),
        total_orders=("total_orders", "sum"),
        late_handovers=("late_handovers", "sum")
    )
)

# Calculate late handover rate
seller_state_analysis["late_rate"] = (
    seller_state_analysis["late_handovers"]
    / seller_state_analysis["total_orders"]
    * 100
)

# Sort by number of late handovers
seller_state_analysis = (
    seller_state_analysis.sort_values(
        "late_handovers",
        ascending=False
    )
)

print("\n--- SELLER PERFORMANCE BY STATE ---")

print(seller_state_analysis.head(10).round(2))


# =====================================
# SELLER HANDOVER VS CUSTOMER REVIEWS
# =====================================

# Create order-level handover data
handover_reviews = (
    order_items[
        ["order_id", "shipping_limit_date"]
    ]
    .copy()
)

# Convert shipping deadline to datetime
handover_reviews["shipping_limit_date"] = (
    pd.to_datetime(
        handover_reviews["shipping_limit_date"]
    )
)

# Get latest shipping deadline per order
handover_reviews = (
    handover_reviews.groupby("order_id")
    ["shipping_limit_date"]
    .max()
    .reset_index()
)

# Merge with orders
handover_reviews = handover_reviews.merge(
    orders[
        [
            "order_id",
            "order_delivered_carrier_date",
            "order_status"
        ]
    ],
    on="order_id",
    how="inner"
)

# Convert carrier date
handover_reviews["order_delivered_carrier_date"] = (
    pd.to_datetime(
        handover_reviews["order_delivered_carrier_date"]
    )
)

# Keep delivered orders with carrier dates
handover_reviews = handover_reviews[
    (handover_reviews["order_status"] == "delivered")
    & (
        handover_reviews[
            "order_delivered_carrier_date"
        ].notna()
    )
].copy()

# Identify late handover
handover_reviews["seller_late"] = (
    handover_reviews["order_delivered_carrier_date"]
    > handover_reviews["shipping_limit_date"]
)

# Merge with final delivery status
handover_reviews = handover_reviews.merge(
    delivered[["order_id", "is_late"]],
    on="order_id",
    how="inner"
)

# Merge with customer reviews
handover_reviews = handover_reviews.merge(
    reviews_per_order,
    on="order_id",
    how="inner"
)

# Analyze customer satisfaction
review_analysis = (
    handover_reviews.groupby(
        ["seller_late", "is_late"]
    )
    .agg(
        total_orders=("order_id", "nunique"),
        avg_review_score=("review_score", "mean")
    )
)

print("\n--- HANDOVER VS CUSTOMER SATISFACTION ---")

print(review_analysis.round(2))


# =====================================
# LOW CUSTOMER RATINGS ANALYSIS
# =====================================

# Identify low ratings
handover_reviews["low_rating"] = (
    handover_reviews["review_score"] <= 2
)

# Analyze low ratings by delivery status
low_rating_analysis = (
    handover_reviews.groupby(
        ["seller_late", "is_late"]
    )
    .agg(
        total_orders=("order_id", "nunique"),
        low_ratings=("low_rating", "sum"),
        low_rating_rate=("low_rating", "mean")
    )
)

# Convert rate to percentage
low_rating_analysis["low_rating_rate"] *= 100

print("\n--- LOW CUSTOMER RATINGS ANALYSIS ---")

print(low_rating_analysis.round(2))


# ====================================
# PREPARE RFM DATA
# ====================================

# Load payments
payments = pd.read_csv(
    "data/olist_order_payments_dataset.csv"
)

# Total payment per order
order_payments = (
    payments.groupby("order_id", as_index=False)
    .agg(total_payment=("payment_value", "sum"))
)

# Merge customers and orders
customer_orders = orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left"
)

# Merge payments
customer_orders = customer_orders.merge(
    order_payments,
    on="order_id",
    how="left"
)

# Keep delivered orders
customer_orders = customer_orders[
    (customer_orders["order_status"] == "delivered")
    & customer_orders["total_payment"].notna()
].copy()

# Convert purchase date
customer_orders["order_purchase_timestamp"] = (
    pd.to_datetime(
        customer_orders["order_purchase_timestamp"]
    )
)

# Set analysis date
snapshot_date = (
    customer_orders["order_purchase_timestamp"].max()
    + pd.Timedelta(days=1)
)

# Calculate RFM
rfm = customer_orders.groupby(
    "customer_unique_id"
).agg(
    last_purchase=("order_purchase_timestamp", "max"),
    frequency=("order_id", "nunique"),
    monetary=("total_payment", "sum")
)

rfm["recency"] = (
    snapshot_date - rfm["last_purchase"]
).dt.days

rfm = rfm[
    ["recency", "frequency", "monetary"]
]

print("\n--- RFM METRICS ---")
print(rfm.head())

# ====================================
# RFM - CUSTOMER BEHAVIOR EXPLORATION
# ====================================

print("\n--- CUSTOMER FREQUENCY DISTRIBUTION ---")

# Number of customers by purchase frequency
frequency_distribution = (
    rfm["frequency"]
    .value_counts()
    .sort_index()
)

print(frequency_distribution.head(15))

# One-time customers
one_time_customers = (
    rfm["frequency"] == 1
).sum()

# Repeat customers
repeat_customers = (
    rfm["frequency"] > 1
).sum()

# Total customers
total_customers = len(rfm)

print("\nTotal customers:", total_customers)

print("\nOne-time customers:", one_time_customers)

print("\nRepeat customers:", repeat_customers)

print("\nOne-time customer percentage:")
print(
    round(
        one_time_customers / total_customers * 100,
        2
    )
)

print("\nRepeat customer percentage:")
print(
    round(
        repeat_customers / total_customers * 100,
        2
    )
)

# Monetary distribution
print("\n--- MONETARY DISTRIBUTION ---")

print(
    rfm["monetary"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]
    ).round(2)
)

# Recency distribution
print("\n--- RECENCY DISTRIBUTION ---")

print(
    rfm["recency"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]
    ).round(2)
)


# ====================================
# RFM SCORING
# ====================================

print("\n--- RFM SCORING ---")

# Recency score
# Lower recency = higher score

rfm["R_score"] = pd.qcut(
    rfm["recency"].rank(method="first"),
    q=5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

# Frequency score
# Adjusted for low repeat purchase rate

def frequency_score(frequency):
    if frequency == 1:
        return 1
    elif frequency == 2:
        return 3
    elif frequency == 3:
        return 4
    else:
        return 5

rfm["F_score"] = (
    rfm["frequency"].apply(frequency_score)
)

# Monetary score
# Higher spending = higher score

rfm["M_score"] = pd.qcut(
    rfm["monetary"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Combined RFM score

rfm["RFM_score"] = (
    rfm["R_score"].astype(str)
    + rfm["F_score"].astype(str)
    + rfm["M_score"].astype(str)
)

print("\nFirst 10 customers:")
print(rfm.head(10))

print("\nRecency score distribution:")
print(rfm["R_score"].value_counts().sort_index())

print("\nFrequency score distribution:")
print(rfm["F_score"].value_counts().sort_index())

print("\nMonetary score distribution:")
print(rfm["M_score"].value_counts().sort_index())


# ====================================
# CUSTOMER SEGMENTATION
# ====================================

print("\n--- CUSTOMER SEGMENTATION ---")

def customer_segment(row):

    r = row["R_score"]
    f = row["F_score"]
    m = row["M_score"]

    # Repeat customers
    if f >= 4 and r >= 4:
        return "Champions"

    elif f >= 3 and r >= 3:
        return "Loyal Customers"

    elif f >= 3 and r <= 2:
        return "At Risk"

    elif f >= 3:
        return "Repeat Customers"

    # One-time customers
    elif r >= 4 and m >= 4:
        return "High Value New"

    elif r >= 4:
        return "Recent Customers"

    elif r <= 2:
        return "Inactive Customers"

    else:
        return "Regular Customers"


# Assign segment
rfm["segment"] = rfm.apply(
    customer_segment,
    axis=1
)

# Analyze segments
segment_analysis = rfm.groupby(
    "segment"
).agg(
    total_customers=("frequency", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
    total_monetary=("monetary", "sum")
)

# Customer percentage
segment_analysis["customer_percentage"] = (
    segment_analysis["total_customers"]
    / len(rfm) * 100
)

# Revenue contribution
segment_analysis["monetary_percentage"] = (
    segment_analysis["total_monetary"]
    / rfm["monetary"].sum() * 100
)

# Sort by total monetary
segment_analysis = segment_analysis.sort_values(
    "total_monetary",
    ascending=False
)

print("\n--- CUSTOMER SEGMENTS ---")

print(segment_analysis.round(2).to_string())

print("\nTotal segmented customers:")
print(segment_analysis["total_customers"].sum())

# Export results
rfm.to_csv(
    "customer_rfm_segments.csv"
)

segment_analysis.to_csv(
    "customer_segment_summary.csv"
)


# =====================================
# CUSTOMER VALUE ANALYSIS
# =====================================

# Calculate order value
order_value = (
    order_items
    .groupby("order_id", as_index=False)
    .agg(
        product_value=("price", "sum"),
        freight_value=("freight_value", "sum")
    )
)

order_value["total_value"] = (
    order_value["product_value"]
    + order_value["freight_value"]
)

# Merge orders with customers
customer_orders = orders.merge(
    customers[
        ["customer_id", "customer_unique_id"]
    ],
    on="customer_id",
    how="inner"
)

# Keep delivered orders
customer_orders = customer_orders[
    customer_orders["order_status"] == "delivered"
].copy()

# Attach order values
customer_orders = customer_orders.merge(
    order_value,
    on="order_id",
    how="inner"
)

# Calculate customer value
customer_value = (
    customer_orders
    .groupby("customer_unique_id")
    .agg(
        total_orders=("order_id", "nunique"),
        total_spent=("total_value", "sum"),
        product_spent=("product_value", "sum"),
        freight_spent=("freight_value", "sum")
    )
)

# Average order value
customer_value["avg_order_value"] = (
    customer_value["total_spent"]
    / customer_value["total_orders"]
)

print("\n--- CUSTOMER VALUE ANALYSIS ---")

print(customer_value.describe().round(2))


# =====================================
# ONE-TIME VS REPEAT CUSTOMERS
# =====================================

customer_value["customer_type"] = (
    customer_value["total_orders"]
    .apply(
        lambda x: "Repeat Customer"
        if x > 1
        else "One-time Customer"
    )
)

customer_type_analysis = (
    customer_value
    .groupby("customer_type")
    .agg(
        total_customers=("total_orders", "count"),
        total_orders=("total_orders", "sum"),
        total_revenue=("total_spent", "sum"),
        avg_customer_value=("total_spent", "mean"),
        avg_order_value=("avg_order_value", "mean")
    )
)

customer_type_analysis["revenue_share"] = (
    customer_type_analysis["total_revenue"]
    / customer_type_analysis["total_revenue"].sum()
    * 100
)

print("\n--- CUSTOMER TYPE ANALYSIS ---")

print(customer_type_analysis.round(2).to_string())


# =====================================
# CUSTOMER REVENUE CONCENTRATION
# =====================================

customer_ranked = customer_value.sort_values(
    "total_spent",
    ascending=False
).copy()

customer_ranked["revenue_share"] = (
    customer_ranked["total_spent"]
    / customer_ranked["total_spent"].sum()
)

customer_ranked["cumulative_revenue"] = (
    customer_ranked["revenue_share"].cumsum()
)

customer_ranked["customer_rank"] = (
    range(1, len(customer_ranked) + 1)
)

customer_ranked["customer_percentage"] = (
    customer_ranked["customer_rank"]
    / len(customer_ranked)
    * 100
)

top_20 = customer_ranked[
    customer_ranked["customer_percentage"] <= 20
]

print("\n--- REVENUE CONCENTRATION ---")

print(
    "Revenue generated by top 20% customers:",
    round(
        top_20["revenue_share"].sum() * 100,
        2
    ),
    "%"
)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.hist(
    customer_value["total_spent"],
    bins=50,
    color="#2E86AB",
    edgecolor="white"
)

plt.title("Customer Value Distribution")
plt.xlabel("Total Customer Spending (BRL)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "customer_value_distribution.png",
    dpi=300
)

plt.show()


# ==========================================
# CUSTOMER COHORT ANALYSIS
# ==========================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare customer purchase history
cohort_data = customer_orders[
    [
        "customer_unique_id",
        "order_id",
        "order_purchase_timestamp"
    ]
].copy()

# Convert purchase date
cohort_data["order_purchase_timestamp"] = (
    pd.to_datetime(
        cohort_data["order_purchase_timestamp"]
    )
)

# Remove missing dates
cohort_data = cohort_data.dropna(
    subset=["order_purchase_timestamp"]
)

# Purchase month
cohort_data["order_month"] = (
    cohort_data["order_purchase_timestamp"]
    .dt.to_period("M")
)

# First purchase month for each customer
cohort_data["cohort_month"] = (
    cohort_data
    .groupby("customer_unique_id")["order_month"]
    .transform("min")
)

# Months since first purchase
cohort_data["cohort_index"] = (
    (cohort_data["order_month"].dt.year
     - cohort_data["cohort_month"].dt.year) * 12
    +
    (cohort_data["order_month"].dt.month
     - cohort_data["cohort_month"].dt.month)
)

# Count unique customers
cohort_counts = (
    cohort_data
    .groupby(["cohort_month", "cohort_index"])
    ["customer_unique_id"]
    .nunique()
    .unstack(fill_value=0)
)

# Calculate retention
cohort_retention = (
    cohort_counts
    .div(cohort_counts[0], axis=0)
    * 100
)

print("\n--- COHORT RETENTION ANALYSIS ---")

print(cohort_retention.round(2).to_string())

# Visualize retention
plt.figure(figsize=(15, 9))

sns.heatmap(
    cohort_retention,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    mask=cohort_counts.eq(0)
)

plt.title("Customer Retention Cohort Analysis")
plt.xlabel("Months Since First Purchase")
plt.ylabel("First Purchase Month")

plt.tight_layout()

plt.savefig(
    "cohort_retention.png",
    dpi=300
)

plt.show()


# ==========================================
# CUSTOMER LIFETIME VALUE ANALYSIS
# ==========================================

print("\n--- CUSTOMER LIFETIME VALUE ANALYSIS ---")

# Historical customer value
clv_data = rfm.copy()

clv_data["historical_value"] = clv_data["monetary"]

# Average order value
clv_data["avg_order_value"] = (
    clv_data["monetary"] /
    clv_data["frequency"]
)

print("\n--- HISTORICAL CUSTOMER VALUE ---")

print(
    clv_data[
        [
            "frequency",
            "monetary",
            "avg_order_value"
        ]
    ].describe().round(2)
)


# ==========================================
# CUSTOMER VALUE BY SEGMENT
# ==========================================

segment_clv = (
    clv_data
    .groupby("segment")
    .agg(
        total_customers=("monetary", "size"),

        avg_historical_value=(
            "historical_value", "mean"
        ),

        avg_order_value=(
            "avg_order_value", "mean"
        ),

        avg_frequency=(
            "frequency", "mean"
        ),

        total_revenue=(
            "monetary", "sum"
        )
    )
    .sort_values(
        "avg_historical_value",
        ascending=False
    )
)

print("\n--- CUSTOMER VALUE BY SEGMENT ---")

print(segment_clv.round(2).to_string())


# ==========================================
# REPEAT PURCHASE BY SEGMENT
# ==========================================

clv_data["is_repeat"] = (
    clv_data["frequency"] > 1
)

repeat_by_segment = (
    clv_data
    .groupby("segment")
    .agg(
        total_customers=("frequency", "size"),

        repeat_customers=(
            "is_repeat", "sum"
        ),

        avg_frequency=(
            "frequency", "mean"
        )
    )
)

repeat_by_segment["repeat_rate"] = (
    repeat_by_segment["repeat_customers"] /
    repeat_by_segment["total_customers"]
) * 100

print("\n--- REPEAT PURCHASE BY SEGMENT ---")

print(repeat_by_segment.round(2).to_string())


# ==========================================
# CUSTOMER VALUE VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

segment_clv["avg_historical_value"].plot(
    kind="bar",
    color="steelblue"
)

plt.title(
    "Average Historical Customer Value by Segment"
)

plt.xlabel("Customer Segment")
plt.ylabel("Average Customer Value (BRL)")

plt.xticks(rotation=35, ha="right")

plt.tight_layout()

plt.savefig(
    "customer_value_by_segment.png",
    dpi=300
)

plt.show()


# ==========================================
# PREDICTIVE CLV - DATA READINESS
# ==========================================

print("\n--- PREDICTIVE CLV DATA READINESS ---")

clv_orders = customer_orders.copy()

clv_orders["order_purchase_timestamp"] = (
    pd.to_datetime(
        clv_orders["order_purchase_timestamp"]
    )
)

print("\nDataset date range:")

print(
    clv_orders["order_purchase_timestamp"]
    .agg(["min", "max"])
)

print("\nTotal unique customers:")

print(
    clv_orders["customer_unique_id"].nunique()
)

print("\nTotal unique orders:")

print(
    clv_orders["order_id"].nunique()
)

print("\nMonthly order distribution:")

monthly_orders = (
    clv_orders
    .groupby(
        clv_orders[
            "order_purchase_timestamp"
        ].dt.to_period("M")
    )["order_id"]
    .nunique()
)

print(monthly_orders.to_string())

# Monthly purchasing customers
monthly_customers = (
    clv_orders
    .groupby(
        clv_orders[
            "order_purchase_timestamp"
        ].dt.to_period("M")
    )["customer_unique_id"]
    .nunique()
)

print("\nMonthly active customers:")

print(monthly_customers.to_string())


# ==========================================
# PREDICTIVE CLV - TEMPORAL MODELING
# ==========================================

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. PREPARE PURCHASE HISTORY
# ==========================================

print("\n--- PREPARING PREDICTION DATA ---")

prediction_orders = orders[
    [
        "order_id",
        "customer_id",
        "order_purchase_timestamp"
    ]
].copy()

prediction_orders["order_purchase_timestamp"] = (
    pd.to_datetime(
        prediction_orders["order_purchase_timestamp"]
    )
)

prediction_orders = prediction_orders.merge(
    customers[
        ["customer_id", "customer_unique_id"]
    ],
    on="customer_id",
    how="inner"
)

# Calculate total order value

prediction_items = order_items.copy()

prediction_items["total_value"] = (
    prediction_items["price"]
    + prediction_items["freight_value"]
)

prediction_values = (
    prediction_items
    .groupby("order_id", as_index=False)
    ["total_value"]
    .sum()
)

prediction_orders = prediction_orders.merge(
    prediction_values,
    on="order_id",
    how="inner"
)

prediction_orders = prediction_orders.dropna(
    subset=[
        "customer_unique_id",
        "order_purchase_timestamp",
        "total_value"
    ]
)

prediction_orders = prediction_orders.drop_duplicates(
    subset=["order_id"]
)

print("Orders:", len(prediction_orders))

print(
    "Customers:",
    prediction_orders["customer_unique_id"].nunique()
)


# ==========================================
# ADD CUSTOMER EXPERIENCE DATA
# ==========================================

print("\n--- PREPARING CUSTOMER EXPERIENCE ---")

# 1. Delivery information

experience_delivery = orders[
    [
        "order_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
].copy()

for col in [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]:
    experience_delivery[col] = pd.to_datetime(
        experience_delivery[col]
    )

experience_delivery["delivery_days"] = (
    experience_delivery["order_delivered_customer_date"]
    - experience_delivery["order_purchase_timestamp"]
).dt.total_seconds() / 86400

experience_delivery["was_late"] = (
    experience_delivery["order_delivered_customer_date"]
    > experience_delivery["order_estimated_delivery_date"]
).astype(float)

# 2. Reviews: retain the date the review became available

experience_reviews = reviews[
    [
        "order_id",
        "review_score",
        "review_answer_timestamp"
    ]
].copy()

experience_reviews["review_answer_timestamp"] = (
    pd.to_datetime(
        experience_reviews["review_answer_timestamp"]
    )
)

# A review is usable only after its answer timestamp.
# Keep individual reviews; aggregate later at each cutoff.

# 3. Freight per order

experience_freight = (
    order_items.groupby("order_id", as_index=False)
    .agg(
        order_freight=("freight_value", "sum")
    )
)

# 4. Product categories

experience_products = pd.read_csv(
    "data/olist_products_dataset.csv",
    usecols=["product_id", "product_category_name"]
)

experience_categories = (
    order_items[
        ["order_id", "product_id"]
    ]
    .merge(
        experience_products,
        on="product_id",
        how="left"
    )
    .groupby("order_id")["product_category_name"]
    .agg(lambda x: set(x.dropna()))
    .reset_index(name="order_categories")
)

# Attach delivery and freight data to the
# prediction history. Do not attach future reviews yet.

prediction_orders = prediction_orders.merge(
    experience_delivery[
        [
            "order_id",
            "order_delivered_customer_date",
            "delivery_days",
            "was_late"
        ]
    ],
    on="order_id",
    how="left",
    validate="one_to_one"
)

prediction_orders = prediction_orders.merge(
    experience_freight,
    on="order_id",
    how="left",
    validate="one_to_one"
)

prediction_orders = prediction_orders.merge(
    experience_categories,
    on="order_id",
    how="left",
    validate="one_to_one"
)

print("Customer experience data prepared!")


# ==========================================
# 2. TEMPORAL CONFIGURATION
# ==========================================

# Every training outcome must be observable
# before the final test snapshot.

HORIZON_DAYS = 120

TRAIN_SNAPSHOTS = [
    pd.Timestamp("2017-07-01"),
    pd.Timestamp("2017-10-01")
]

TEST_SNAPSHOT = pd.Timestamp("2018-03-01")

TOP_K_VALUES = [
    10, 25, 50, 100, 200, 500, 1000
]

FEATURE_COLUMNS = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "customer_tenure",
    "avg_days_between_orders",
    "is_one_time",
    "orders_last_30d",
    "orders_last_90d",
    "avg_review_score",
    "late_delivery_rate",
    "avg_delivery_days",
    "avg_freight",
    "product_category_diversity",
    "has_review",
    "has_completed_delivery"
]

# October 1 + 120 days is before March 1.
# Therefore training labels are known
# before the final test snapshot.

for snapshot_date in TRAIN_SNAPSHOTS:

    label_end = (
        snapshot_date
        + pd.Timedelta(days=HORIZON_DAYS)
    )

    if label_end > TEST_SNAPSHOT:
        raise ValueError(
            "Training labels overlap the test period."
        )

# Ensure the test observation window exists.

test_end = (
    TEST_SNAPSHOT
    + pd.Timedelta(days=HORIZON_DAYS)
)

if (
    prediction_orders[
        "order_purchase_timestamp"
    ].max() < test_end
):
    raise ValueError(
        "The dataset does not cover the full "
        "future observation window."
    )


# ==========================================
# 3. CUSTOMER FEATURE ENGINEERING
# ==========================================

def build_customer_snapshot(
    purchase_history,
    cutoff_date
):

    past = purchase_history[
        purchase_history[
            "order_purchase_timestamp"
        ] < cutoff_date
    ].copy()

    future_end = (
        cutoff_date
        + pd.Timedelta(days=HORIZON_DAYS)
    )

    future = purchase_history[
        (
            purchase_history[
                "order_purchase_timestamp"
            ] >= cutoff_date
        )
        &
        (
            purchase_history[
                "order_purchase_timestamp"
            ] < future_end
        )
    ].copy()

    # Historical customer features

    features = (
        past
        .groupby("customer_unique_id")
        .agg(
            first_purchase=(
                "order_purchase_timestamp",
                "min"
            ),
            last_purchase=(
                "order_purchase_timestamp",
                "max"
            ),
            frequency=(
                "order_id",
                "nunique"
            ),
            monetary=(
                "total_value",
                "sum"
            )
        )
    )

    # Recency

    features["recency"] = (
        cutoff_date
        - features["last_purchase"]
    ).dt.total_seconds() / 86400

    # Customer tenure

    features["customer_tenure"] = (
        cutoff_date
        - features["first_purchase"]
    ).dt.total_seconds() / 86400

    # Average order value

    features["avg_order_value"] = (
        features["monetary"]
        / features["frequency"]
    )

    # One-time customer indicator

    features["is_one_time"] = (
        features["frequency"] == 1
    ).astype(int)

    # Average purchase interval

    features["avg_days_between_orders"] = (
        (
            features["last_purchase"]
            - features["first_purchase"]
        ).dt.total_seconds() / 86400
        /
        (
            features["frequency"] - 1
        ).replace(0, np.nan)
    )

    # Missing interval means no repeat
    # interval has been observed yet.

    features["avg_days_between_orders"] = (
        features["avg_days_between_orders"]
        .fillna(-1)
    )

    # Last 30 days

    last_30 = past[
        past["order_purchase_timestamp"]
        >= cutoff_date - pd.Timedelta(days=30)
    ]

    features["orders_last_30d"] = (
        last_30
        .groupby("customer_unique_id")
        ["order_id"]
        .nunique()
        .reindex(
            features.index,
            fill_value=0
        )
    )

    # Last 90 days

    last_90 = past[
        past["order_purchase_timestamp"]
        >= cutoff_date - pd.Timedelta(days=90)
    ]

    features["orders_last_90d"] = (
        last_90
        .groupby("customer_unique_id")
        ["order_id"]
        .nunique()
        .reindex(
            features.index,
            fill_value=0
        )
    )

    # ======================================
    # CUSTOMER EXPERIENCE FEATURES
    # ======================================

    # Only deliveries completed before the
    # prediction cutoff are observable.

    completed = past[
        past["order_delivered_customer_date"].notna()
        &
        (
            past["order_delivered_customer_date"]
            < cutoff_date
        )
    ].copy()

    completed = completed[
        completed["delivery_days"] >= 0
    ].copy()

    delivery_features = (
        completed.groupby("customer_unique_id")
        .agg(
            late_delivery_rate=(
                "was_late", "mean"
            ),
            avg_delivery_days=(
                "delivery_days", "mean"
            )
        )
    )

    features = features.join(
        delivery_features
    )

    # Review scores are usable only if
    # submitted before the prediction cutoff.

    available_reviews = experience_reviews[
        experience_reviews[
            "review_answer_timestamp"
        ].notna()
        &
        (
            experience_reviews[
                "review_answer_timestamp"
            ] < cutoff_date
        )
    ].copy()

    customer_reviews = past[
        ["order_id", "customer_unique_id"]
    ].merge(
        available_reviews[
            ["order_id", "review_score"]
        ],
        on="order_id",
        how="inner"
    )

    review_features = (
        customer_reviews
        .groupby("customer_unique_id")
        .agg(
            avg_review_score=(
                "review_score", "mean"
            )
        )
    )

    features = features.join(
        review_features
    )

    # Average historical freight

    freight_features = (
        past.groupby("customer_unique_id")
        .agg(
            avg_freight=(
                "order_freight", "mean"
            )
        )
    )

    features = features.join(
        freight_features
    )

    # Number of distinct product categories
    # across the customer's historical orders.

    category_features = (
        past.groupby("customer_unique_id")
        ["order_categories"]
        .agg(
            lambda category_sets: len(
                set().union(
                    *[
                        x for x in category_sets
                        if isinstance(x, set)
                    ]
                )
            )
        )
    )

    features["product_category_diversity"] = (
        category_features
    )

    # Missing delivery/review data means
    # the information was not yet observed.
    # Add indicators to distinguish missing
    # values from actual zeros.

    features["has_review"] = (
        features["avg_review_score"].notna()
    ).astype(int)

    features["has_completed_delivery"] = (
        features["avg_delivery_days"].notna()
    ).astype(int)

    features["avg_review_score"] = (
        features["avg_review_score"].fillna(-1)
    )

    features["late_delivery_rate"] = (
        features["late_delivery_rate"].fillna(-1)
    )

    features["avg_delivery_days"] = (
        features["avg_delivery_days"].fillna(-1)
    )

    features["avg_freight"] = (
        features["avg_freight"].fillna(-1)
    )

    features["product_category_diversity"] = (
        features["product_category_diversity"]
        .fillna(0)
    )

    # Target: another purchase within 120 days

    future_customers = set(
        future["customer_unique_id"]
    )

    features["will_purchase_again"] = (
        features.index.isin(
            future_customers
        )
        .astype(int)
    )

    features["snapshot_date"] = cutoff_date

    features = features.reset_index()

    print(
        f"\nSnapshot: {cutoff_date.date()}"
    )

    print("Customers:", len(features))

    print(
        "Returning customers:",
        features["will_purchase_again"].sum()
    )

    print(
        "Return rate:",
        round(
            features[
                "will_purchase_again"
            ].mean() * 100,
            2
        ),
        "%"
    )

    return features


# ==========================================
# 4. BUILD TRAINING AND TEST DATA
# ==========================================

print("\n--- TEMPORAL DATA SPLIT ---")

training_snapshots = []

for snapshot_date in TRAIN_SNAPSHOTS:

    snapshot = build_customer_snapshot(
        prediction_orders,
        snapshot_date
    )

    training_snapshots.append(snapshot)

train_data = pd.concat(
    training_snapshots,
    ignore_index=True
)

test_data = build_customer_snapshot(
    prediction_orders,
    TEST_SNAPSHOT
)

# Customers can appear in multiple snapshots.
# These are separate customer-time observations.

print("\nTraining observations:", len(train_data))

print("Testing customers:", len(test_data))

print(
    "\nTraining positive rate:",
    round(
        train_data[
            "will_purchase_again"
        ].mean() * 100,
        2
    ),
    "%"
)

print(
    "Testing positive rate:",
    round(
        test_data[
            "will_purchase_again"
        ].mean() * 100,
        2
    ),
    "%"
)

X_train = train_data[
    FEATURE_COLUMNS
].copy()

y_train = train_data[
    "will_purchase_again"
].copy()

X_test = test_data[
    FEATURE_COLUMNS
].copy()

y_test = test_data[
    "will_purchase_again"
].copy()

if y_train.nunique() < 2:
    raise ValueError(
        "Training data contains only one class."
    )


# ==========================================
# 5. EVALUATION FUNCTIONS
# ==========================================

def evaluate_prediction(
    actual,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "positive_rate": actual.mean(),

        "average_precision":
            average_precision_score(
                actual,
                probabilities
            ),

        "roc_auc":
            roc_auc_score(
                actual,
                probabilities
            )
            if actual.nunique() == 2
            else np.nan,

        "precision":
            precision_score(
                actual,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                actual,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                actual,
                predictions,
                zero_division=0
            ),

        "predicted_positive":
            predictions.sum()
    }

    return metrics


def calculate_topk(
    actual,
    probabilities
):

    ranked = pd.DataFrame({
        "actual": np.asarray(actual),
        "probability": np.asarray(probabilities)
    })

    ranked = ranked.sort_values(
        "probability",
        ascending=False
    )

    baseline = ranked["actual"].mean()

    total_returning = ranked["actual"].sum()

    results = []

    for k in TOP_K_VALUES:

        if k > len(ranked):
            continue

        selected = ranked.head(k)

        returning = selected["actual"].sum()

        precision = returning / k

        recall = (
            returning / total_returning
            if total_returning > 0
            else np.nan
        )

        lift = (
            precision / baseline
            if baseline > 0
            else np.nan
        )

        results.append({
            "Top_K": k,
            "Returning": returning,
            "Precision": precision,
            "Recall": recall,
            "Lift": lift
        })

    return pd.DataFrame(results)


# ==========================================
# 6. BUILD MACHINE LEARNING MODELS
# ==========================================

print("\n--- BUILDING MODELS ---")

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# Add XGBoost if installed

try:

    from xgboost import XGBClassifier

    positive_count = y_train.sum()

    negative_count = (
        len(y_train) - positive_count
    )

    class_ratio = (
        negative_count / positive_count
    )

    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=class_ratio,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    print("XGBoost is available.")

except ImportError:

    print(
        "XGBoost is not installed."
    )

    print(
        "Install it with: pip install xgboost"
    )


# ==========================================
# 7. TRAIN AND COMPARE MODELS
# ==========================================

output_folder = Path(
    "prediction_outputs"
)

output_folder.mkdir(
    exist_ok=True
)

comparison_results = []

for model_name, model_object in models.items():

    print(
        f"\n========== {model_name} =========="
    )

    # Train

    model_object.fit(
        X_train,
        y_train
    )

    # Predict

    probabilities = (
        model_object.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # Classification report

    print(
        "\n--- CLASSIFICATION REPORT ---"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Confusion matrix

    print(
        "\n--- CONFUSION MATRIX ---"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # Metrics

    metrics = evaluate_prediction(
        y_test,
        probabilities
    )

    print(
        "\n--- EVALUATION METRICS ---"
    )

    for metric_name, value in metrics.items():

        print(
            f"{metric_name}: {value:.4f}"
        )

    comparison_results.append({
        "Model": model_name,
        **metrics
    })

    # Top-K analysis

    topk_results = calculate_topk(
        y_test,
        probabilities
    )

    print(
        "\n--- TOP-K ANALYSIS ---"
    )

    print(
        topk_results.round(4).to_string(
            index=False
        )
    )

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    topk_results.to_csv(
        output_folder /
        f"{safe_name}_topk.csv",
        index=False
    )

    # Save customer scores

    customer_scores = test_data[
        [
            "customer_unique_id",
            "will_purchase_again"
        ]
    ].copy()

    customer_scores[
        "predicted_probability"
    ] = probabilities

    customer_scores = (
        customer_scores.sort_values(
            "predicted_probability",
            ascending=False
        )
    )

    customer_scores.to_csv(
        output_folder /
        f"{safe_name}_customer_scores.csv",
        index=False
    )

    # Feature importance

    if hasattr(
        model_object,
        "feature_importances_"
    ):

        importance = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance":
                model_object.feature_importances_
        })

        importance = importance.sort_values(
            "importance",
            ascending=False
        )

        print(
            "\n--- FEATURE IMPORTANCE ---"
        )

        print(
            importance.to_string(
                index=False
            )
        )

        importance.to_csv(
            output_folder /
            f"{safe_name}_importance.csv",
            index=False
        )


# ==========================================
# 8. FINAL MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame(
    comparison_results
)

comparison = comparison.sort_values(
    "average_precision",
    ascending=False
)

print(
    "\n========== OUT-OF-TIME COMPARISON =========="
)

print(
    comparison.round(4).to_string(
        index=False
    )
)

comparison.to_csv(
    output_folder /
    "model_comparison.csv",
    index=False
)

print(
    "\nPrediction analysis completed!"
)

print(
    "Results saved in:",
    output_folder.resolve()
)

print(
    "\nNOTE: This model predicts whether "
    "a customer will purchase again "
    "within 120 days. It does not "
    "directly predict future CLV."
)

print(
    "\nIMPORTANT: Historical purchase "
    "probabilities do not establish "
    "the incremental impact of "
    "a marketing campaign."
)

from sklearn.base import clone

feature_groups = {
    "RFM Only": [
        "recency",
        "frequency",
        "monetary"
    ],

    "Purchase Behavior": [
        "recency",
        "frequency",
        "monetary",
        "avg_order_value",
        "customer_tenure",
        "avg_days_between_orders",
        "is_one_time",
        "orders_last_30d",
        "orders_last_90d"
    ],

    "Purchase + Experience": FEATURE_COLUMNS
}

ablation_results = []

for group_name, columns in feature_groups.items():

    model = clone(models["Logistic Regression"])

    model.fit(
        train_data[columns],
        y_train
    )

    probabilities = model.predict_proba(
        test_data[columns]
    )[:, 1]

    metrics = evaluate_prediction(
        y_test,
        probabilities
    )

    topk = calculate_topk(
        y_test,
        probabilities
    )

    top500 = topk.loc[
        topk["Top_K"] == 500
    ]

    ablation_results.append({
        "Feature_Group": group_name,
        "PR_AUC": metrics["average_precision"],
        "ROC_AUC": metrics["roc_auc"],
        "Top_500_Returning": (
            int(top500["Returning"].iloc[0])
            if not top500.empty else np.nan
        ),
        "Top_500_Lift": (
            float(top500["Lift"].iloc[0])
            if not top500.empty else np.nan
        )
    })

ablation_comparison = pd.DataFrame(
    ablation_results
)

print(
    ablation_comparison.round(4).to_string(
        index=False
    )
)


# ==========================================
# FULL MODEL ABLATION STUDY
# ==========================================

from sklearn.base import clone

print("\n========== FULL ABLATION STUDY ==========")

feature_groups = {
    "RFM Only": [
        "recency",
        "frequency",
        "monetary"
    ],

    "Purchase Behavior": [
        "recency",
        "frequency",
        "monetary",
        "avg_order_value",
        "customer_tenure",
        "avg_days_between_orders",
        "is_one_time",
        "orders_last_30d",
        "orders_last_90d"
    ],

    "Purchase + Experience": FEATURE_COLUMNS
}

ablation_results = []

for model_name, base_model in models.items():

    for group_name, columns in feature_groups.items():

        print(
            f"\n--- {model_name} | {group_name} ---"
        )

        # Create a fresh model
        model = clone(base_model)

        # Train using selected features
        model.fit(
            train_data[columns],
            y_train
        )

        # Predict probabilities
        probabilities = model.predict_proba(
            test_data[columns]
        )[:, 1]

        # Calculate evaluation metrics
        metrics = evaluate_prediction(
            y_test,
            probabilities
        )

        # Calculate Top-K performance
        topk = calculate_topk(
            y_test,
            probabilities
        )

        # Extract Top-500 results
        top500 = topk.loc[
            topk["Top_K"] == 500
        ]

        # Extract Top-1000 results
        top1000 = topk.loc[
            topk["Top_K"] == 1000
        ]

        ablation_results.append({

            "Model": model_name,

            "Feature_Group": group_name,

            "PR_AUC": metrics["average_precision"],

            "ROC_AUC": metrics["roc_auc"],

            "Top_500_Returning": (
                int(top500["Returning"].iloc[0])
                if not top500.empty else np.nan
            ),

            "Top_500_Lift": (
                float(top500["Lift"].iloc[0])
                if not top500.empty else np.nan
            ),

            "Top_1000_Returning": (
                int(top1000["Returning"].iloc[0])
                if not top1000.empty else np.nan
            ),

            "Top_1000_Lift": (
                float(top1000["Lift"].iloc[0])
                if not top1000.empty else np.nan
            )

        })


# ==========================================
# FINAL COMPARISON
# ==========================================

ablation_comparison = pd.DataFrame(
    ablation_results
)

ablation_comparison = (
    ablation_comparison.sort_values(
        "PR_AUC",
        ascending=False
    )
)

print(
    "\n========== FULL ABLATION COMPARISON =========="
)

print(
    ablation_comparison.round(4).to_string(
        index=False
    )
)


# ==========================================
# SAVE RESULTS
# ==========================================

ablation_comparison.to_csv(
    output_folder / "full_ablation_comparison.csv",
    index=False
)

print(
    "\nFull ablation study completed!"
)


# ==========================================
# TEMPORAL ROBUSTNESS TEST
# ==========================================

from sklearn.base import clone
import pandas as pd
import numpy as np

print("\n========== TEMPORAL ROBUSTNESS TEST ==========")

# Additional historical snapshots
candidate_snapshots = [
    pd.Timestamp("2017-04-01"),
    pd.Timestamp("2017-07-01"),
    pd.Timestamp("2017-10-01"),
    pd.Timestamp("2018-01-01"),
    pd.Timestamp("2018-03-01"),
    pd.Timestamp("2018-05-01")
]

test_snapshots = [
    pd.Timestamp("2018-01-01"),
    pd.Timestamp("2018-03-01"),
    pd.Timestamp("2018-05-01")
]

# Use the latest available purchase timestamp
last_available_date = (
    prediction_orders["order_purchase_timestamp"].max()
)

# Build all snapshots that have a complete future window
snapshot_cache = {}

for snapshot_date in candidate_snapshots:

    future_end = (
        snapshot_date
        + pd.Timedelta(days=HORIZON_DAYS)
    )

    if last_available_date < future_end:
        print(
            f"Skipping {snapshot_date.date()}: "
            "incomplete future window"
        )
        continue

    snapshot_cache[snapshot_date] = (
        build_customer_snapshot(
            prediction_orders,
            snapshot_date
        )
    )


# ==========================================
# FEATURE GROUPS
# ==========================================

robustness_features = {
    "RFM Only": [
        "recency",
        "frequency",
        "monetary"
    ],

    "Purchase + Experience": FEATURE_COLUMNS
}


# ==========================================
# TRAIN AND EVALUATE
# ==========================================

robustness_results = []

for test_date in test_snapshots:

    if test_date not in snapshot_cache:
        continue

    # Only use training snapshots whose
    # 120-day outcomes are fully known
    # before the test snapshot.

    eligible_train_dates = [
        date
        for date in candidate_snapshots
        if (
            date in snapshot_cache
            and date + pd.Timedelta(
                days=HORIZON_DAYS
            ) <= test_date
        )
    ]

    if not eligible_train_dates:
        print(
            f"No eligible training data "
            f"for {test_date.date()}"
        )
        continue

    temporal_train = pd.concat(
        [
            snapshot_cache[date]
            for date in eligible_train_dates
        ],
        ignore_index=True
    )

    temporal_test = (
        snapshot_cache[test_date].copy()
    )

    y_temporal_train = (
        temporal_train["will_purchase_again"]
    )

    y_temporal_test = (
        temporal_test["will_purchase_again"]
    )

    if y_temporal_train.nunique() < 2:
        continue

    print(
        f"\n--- Test Snapshot: {test_date.date()} ---"
    )

    print(
        "Training snapshots:",
        [
            d.date()
            for d in eligible_train_dates
        ]
    )

    print(
        "Training observations:",
        len(temporal_train)
    )

    print(
        "Testing customers:",
        len(temporal_test)
    )

    print(
        "Testing return rate:",
        round(
            y_temporal_test.mean() * 100,
            2
        ),
        "%"
    )

    for group_name, columns in robustness_features.items():

        model = clone(
            models["Logistic Regression"]
        )

        model.fit(
            temporal_train[columns],
            y_temporal_train
        )

        probabilities = model.predict_proba(
            temporal_test[columns]
        )[:, 1]

        metrics = evaluate_prediction(
            y_temporal_test,
            probabilities
        )

        topk = calculate_topk(
            y_temporal_test,
            probabilities
        )

        top500 = topk[
            topk["Top_K"] == 500
        ]

        robustness_results.append({

            "Test_Snapshot": test_date.date(),

            "Feature_Group": group_name,

            "Training_Observations": len(
                temporal_train
            ),

            "Testing_Customers": len(
                temporal_test
            ),

            "Positive_Rate": metrics[
                "positive_rate"
            ],

            "PR_AUC": metrics[
                "average_precision"
            ],

            "ROC_AUC": metrics[
                "roc_auc"
            ],

            "Top_500_Returning": (
                int(
                    top500["Returning"].iloc[0]
                )
                if not top500.empty
                else np.nan
            ),

            "Top_500_Lift": (
                float(
                    top500["Lift"].iloc[0]
                )
                if not top500.empty
                else np.nan
            )

        })


# ==========================================
# FINAL ROBUSTNESS COMPARISON
# ==========================================

robustness_comparison = pd.DataFrame(
    robustness_results
)

if robustness_comparison.empty:

    print(
        "\nNo valid temporal tests were available."
    )

else:

    print(
        "\n========== TEMPORAL ROBUSTNESS COMPARISON =========="
    )

    print(
        robustness_comparison.round(4).to_string(
            index=False
        )
    )

    robustness_comparison.to_csv(
        output_folder /
        "temporal_robustness_comparison.csv",
        index=False
    )

    print(
        "\nTemporal robustness test completed!"
    )


    # ==========================================
# ACTIVITY WINDOW EXPERIMENT
# ==========================================

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import average_precision_score


print("\n========== ACTIVITY WINDOW EXPERIMENT ==========")


# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================

DEV_SNAPSHOT = pd.Timestamp("2018-03-01")

ACTIVITY_TRAIN_SNAPSHOTS = [
    pd.Timestamp("2017-04-01"),
    pd.Timestamp("2017-07-01"),
    pd.Timestamp("2017-10-01")
]

ACTIVITY_WINDOWS = {
    "All History": None,
    "180 Days": 180,
    "90 Days": 90
}

TOP_K_PERCENTAGES = [1, 2, 5, 10]


# ==========================================
# 2. BUILD TRAINING SNAPSHOTS
# ==========================================

training_snapshots = []

for snapshot_date in ACTIVITY_TRAIN_SNAPSHOTS:

    # Training labels must be fully observable
    # before the development snapshot.

    label_end = (
        snapshot_date
        + pd.Timedelta(days=HORIZON_DAYS)
    )

    if label_end > DEV_SNAPSHOT:
        raise ValueError(
            "Training labels overlap the Dev period."
        )

    snapshot = build_customer_snapshot(
        prediction_orders,
        snapshot_date
    )

    training_snapshots.append(snapshot)


train_all = pd.concat(
    training_snapshots,
    ignore_index=True
)


# ==========================================
# 3. BUILD FIXED DEV SET
# ==========================================

dev_data = build_customer_snapshot(
    prediction_orders,
    DEV_SNAPSHOT
)

X_dev = dev_data[FEATURE_COLUMNS].copy()

y_dev = dev_data["will_purchase_again"].copy()


print("\nFixed Dev customers:", len(dev_data))

print(
    "Dev positive rate:",
    round(y_dev.mean() * 100, 4),
    "%"
)


# ==========================================
# 4. ACTIVITY FILTER
# ==========================================

def filter_active_customers(
    dataset,
    window_days
):

    if window_days is None:
        return dataset.copy()

    # Recency was calculated relative
    # to each row's own snapshot date.

    filtered = dataset[
        dataset["recency"] <= window_days
    ].copy()

    return filtered


# ==========================================
# 5. PRECISION AT TOP K%
# ==========================================

def precision_at_top_percent(
    actual,
    probabilities,
    percentages
):

    ranked = pd.DataFrame({
        "actual": np.asarray(actual),
        "probability": np.asarray(probabilities)
    })

    ranked = ranked.sort_values(
        "probability",
        ascending=False
    )

    baseline = ranked["actual"].mean()

    results = {}

    for percentage in percentages:

        k = max(
            1,
            int(
                np.ceil(
                    len(ranked)
                    * percentage / 100
                )
            )
        )

        selected = ranked.head(k)

        returning = int(
            selected["actual"].sum()
        )

        precision = returning / k

        lift = (
            precision / baseline
            if baseline > 0
            else np.nan
        )

        results[
            f"Precision_Top_{percentage}pct"
        ] = precision

        results[
            f"Returning_Top_{percentage}pct"
        ] = returning

        results[
            f"Lift_Top_{percentage}pct"
        ] = lift

    return results


# ==========================================
# 6. DEFINE IDENTICAL MODEL
# ==========================================

base_model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    )

])


# ==========================================
# 7. TRAIN AND EVALUATE
# ==========================================

experiment_results = []

dev_predictions = dev_data[
    ["customer_unique_id", "will_purchase_again"]
].copy()

for experiment_name, window_days in ACTIVITY_WINDOWS.items():

    print(
        f"\n========== {experiment_name} =========="
    )

    # Filter training observations only.

    train_filtered = filter_active_customers(
        train_all,
        window_days
    )

    X_train_exp = train_filtered[
        FEATURE_COLUMNS
    ].copy()

    y_train_exp = train_filtered[
        "will_purchase_again"
    ].copy()

    if y_train_exp.nunique() < 2:
        raise ValueError(
            f"{experiment_name}: only one class."
        )

    print(
        "Training observations:",
        len(train_filtered)
    )

    print(
        "Training positive rate:",
        round(
            y_train_exp.mean() * 100,
            4
        ),
        "%"
    )

    # Clone ensures a fresh model
    # for every experiment.

    model = clone(base_model)

    model.fit(
        X_train_exp,
        y_train_exp
    )

    # Same Dev customers for all models.

    probabilities = model.predict_proba(
        X_dev
    )[:, 1]

    pr_auc = average_precision_score(
        y_dev,
        probabilities
    )

    topk_metrics = precision_at_top_percent(
        y_dev,
        probabilities,
        TOP_K_PERCENTAGES
    )

    result = {
        "Experiment": experiment_name,
        "Train_Observations": len(train_filtered),
        "Train_Positive_Rate": y_train_exp.mean(),
        "Dev_Customers": len(dev_data),
        "Dev_Positive_Rate": y_dev.mean(),
        "PR_AUC": pr_auc,
        **topk_metrics
    }

    experiment_results.append(result)

    print("PR-AUC:", round(pr_auc, 5))

    for percentage in TOP_K_PERCENTAGES:

        precision = topk_metrics[
            f"Precision_Top_{percentage}pct"
        ]

        returning = topk_metrics[
            f"Returning_Top_{percentage}pct"
        ]

        lift = topk_metrics[
            f"Lift_Top_{percentage}pct"
        ]

        print(
            f"Top {percentage}%: "
            f"Returning={returning}, "
            f"Precision={precision:.4f}, "
            f"Lift={lift:.2f}"
        )

    # Save scores for comparison.

    safe_name = (
        experiment_name.lower()
        .replace(" ", "_")
    )

    dev_predictions[
        f"score_{safe_name}"
    ] = probabilities


# ==========================================
# 8. FINAL COMPARISON
# ==========================================

activity_comparison = pd.DataFrame(
    experiment_results
)

activity_comparison = (
    activity_comparison.sort_values(
        "PR_AUC",
        ascending=False
    )
)

display_columns = [
    "Experiment",
    "Train_Observations",
    "Train_Positive_Rate",
    "PR_AUC",
    "Precision_Top_1pct",
    "Precision_Top_2pct",
    "Precision_Top_5pct",
    "Precision_Top_10pct"
]

print(
    "\n========== ACTIVITY WINDOW COMPARISON =========="
)

print(
    activity_comparison[
        display_columns
    ].round(5).to_string(index=False)
)


# ==========================================
# 9. EXPORT RESULTS
# ==========================================

activity_comparison.to_csv(
    output_folder /
    "activity_window_comparison.csv",
    index=False
)

dev_predictions.to_csv(
    output_folder /
    "activity_window_dev_predictions.csv",
    index=False
)

print("\nActivity window experiment completed!")


# ==========================================
# RFM OBSERVATION WINDOW EXPERIMENT
# ==========================================

from sklearn.base import clone
from sklearn.metrics import average_precision_score

print("\n========== RFM WINDOW EXPERIMENT ==========")


# ==========================================
# 1. CONFIGURATION
# ==========================================

RFM_WINDOWS = {
    "All History": None,
    "180 Days": 180,
    "90 Days": 90
}

RFM_TRAIN_SNAPSHOTS = [
    pd.Timestamp("2017-04-01"),
    pd.Timestamp("2017-07-01"),
    pd.Timestamp("2017-10-01")
]

RFM_DEV_SNAPSHOT = pd.Timestamp("2018-03-01")

RFM_TOP_K = [1, 2, 5, 10]


# ==========================================
# 2. BUILD WINDOW FEATURES
# ==========================================

def build_window_snapshot(
    cutoff_date,
    window_days
):

    # Start with the existing snapshot.
    # This preserves the same customers,
    # labels and experience features.

    snapshot = build_customer_snapshot(
        prediction_orders,
        cutoff_date
    )

    all_past = prediction_orders[
        prediction_orders[
            "order_purchase_timestamp"
        ] < cutoff_date
    ].copy()

    if window_days is None:

        window_past = all_past.copy()

    else:

        window_start = (
            cutoff_date
            - pd.Timedelta(days=window_days)
        )

        window_past = all_past[
            all_past[
                "order_purchase_timestamp"
            ] >= window_start
        ].copy()

    # ======================================
    # WINDOW-BASED PURCHASE FEATURES
    # ======================================

    window_features = (
        window_past
        .groupby("customer_unique_id")
        .agg(
            window_first_purchase=(
                "order_purchase_timestamp",
                "min"
            ),

            window_last_purchase=(
                "order_purchase_timestamp",
                "max"
            ),

            window_frequency=(
                "order_id",
                "nunique"
            ),

            window_monetary=(
                "total_value",
                "sum"
            )
        )
    )

    # Average order value

    window_features["window_avg_order_value"] = (
        window_features["window_monetary"]
        / window_features["window_frequency"]
    )

    # Purchase interval

    window_features["window_avg_days_between_orders"] = (
        (
            window_features["window_last_purchase"]
            - window_features["window_first_purchase"]
        ).dt.total_seconds() / 86400
        /
        (
            window_features["window_frequency"] - 1
        ).replace(0, np.nan)
    )

    # Recency relative to cutoff

    window_features["window_recency"] = (
        cutoff_date
        - window_features["window_last_purchase"]
    ).dt.total_seconds() / 86400

    # ======================================
    # JOIN WINDOW FEATURES
    # ======================================

    snapshot = snapshot.merge(
        window_features,
        on="customer_unique_id",
        how="left",
        validate="one_to_one"
    )

    # Customers without purchases in the
    # window remain in the dataset.

    snapshot["has_window_purchase"] = (
        snapshot["window_frequency"].notna()
    ).astype(int)

    # Existing customers whose purchases
    # are entirely outside the window.

    snapshot["has_older_history"] = (
        (
            snapshot["has_window_purchase"] == 0
        )
        &
        (
            snapshot["frequency"] > 0
        )
    ).astype(int)

    # ======================================
    # REPLACE RFM FEATURES
    # ======================================

    snapshot["frequency"] = (
        snapshot["window_frequency"].fillna(0)
    )

    snapshot["monetary"] = (
        snapshot["window_monetary"].fillna(0)
    )

    snapshot["avg_order_value"] = (
        snapshot["window_avg_order_value"]
        .fillna(0)
    )

    snapshot["avg_days_between_orders"] = (
        snapshot["window_avg_days_between_orders"]
        .fillna(-1)
    )

    # No purchase in the window:
    # recency is capped at the window length.

    if window_days is not None:

        snapshot["recency"] = (
            snapshot["window_recency"]
            .fillna(window_days)
        )

    else:

        snapshot["recency"] = (
            snapshot["window_recency"]
        )

    snapshot["is_one_time"] = (
        snapshot["frequency"] == 1
    ).astype(int)

    # ======================================
    # WINDOW-BASED RECENT PURCHASE COUNTS
    # ======================================

    # These are already calculated relative
    # to the snapshot date, so we retain them.

    # ======================================
    # CLEAN TEMPORARY COLUMNS
    # ======================================

    snapshot = snapshot.drop(
        columns=[
            "window_first_purchase",
            "window_last_purchase",
            "window_frequency",
            "window_monetary",
            "window_avg_order_value",
            "window_avg_days_between_orders",
            "window_recency"
        ]
    )

    return snapshot


# ==========================================
# 3. MODEL FEATURES
# ==========================================

WINDOW_FEATURE_COLUMNS = (
    FEATURE_COLUMNS
    + [
        "has_window_purchase",
        "has_older_history"
    ]
)


# ==========================================
# 4. TRAIN AND EVALUATE
# ==========================================

window_results = []

for window_name, window_days in RFM_WINDOWS.items():

    print(
        f"\n========== {window_name} =========="
    )

    training_parts = []

    for snapshot_date in RFM_TRAIN_SNAPSHOTS:

        label_end = (
            snapshot_date
            + pd.Timedelta(days=HORIZON_DAYS)
        )

        if label_end > RFM_DEV_SNAPSHOT:
            raise ValueError(
                "Training labels overlap Dev."
            )

        training_parts.append(
            build_window_snapshot(
                snapshot_date,
                window_days
            )
        )

    train_window = pd.concat(
        training_parts,
        ignore_index=True
    )

    dev_window = build_window_snapshot(
        RFM_DEV_SNAPSHOT,
        window_days
    )

    X_train_window = train_window[
        WINDOW_FEATURE_COLUMNS
    ].copy()

    y_train_window = train_window[
        "will_purchase_again"
    ].copy()

    X_dev_window = dev_window[
        WINDOW_FEATURE_COLUMNS
    ].copy()

    y_dev_window = dev_window[
        "will_purchase_again"
    ].copy()

    if X_train_window.isna().any().any():
        raise ValueError(
            "Missing training features."
        )

    if X_dev_window.isna().any().any():
        raise ValueError(
            "Missing Dev features."
        )

    model = clone(base_model)

    model.fit(
        X_train_window,
        y_train_window
    )

    probabilities = model.predict_proba(
        X_dev_window
    )[:, 1]

    pr_auc = average_precision_score(
        y_dev_window,
        probabilities
    )

    topk = precision_at_top_percent(
        y_dev_window,
        probabilities,
        RFM_TOP_K
    )

    result = {
        "Window": window_name,
        "Train_Observations": len(train_window),
        "Dev_Customers": len(dev_window),
        "PR_AUC": pr_auc,
        **topk
    }

    window_results.append(result)

    print(
        "Training observations:",
        len(train_window)
    )

    print(
        "Dev customers:",
        len(dev_window)
    )

    print(
        "PR-AUC:",
        round(pr_auc, 5)
    )

    for percentage in RFM_TOP_K:

        print(
            f"Top {percentage}%:",
            "Precision =",
            round(
                topk[
                    f"Precision_Top_{percentage}pct"
                ],
                5
            ),
            "| Returning =",
            topk[
                f"Returning_Top_{percentage}pct"
            ]
        )


# ==========================================
# 5. COMPARISON
# ==========================================

window_comparison = pd.DataFrame(
    window_results
)

window_comparison = (
    window_comparison.sort_values(
        "PR_AUC",
        ascending=False
    )
)

print(
    "\n========== RFM WINDOW COMPARISON =========="
)

comparison_columns = [
    "Window",
    "Train_Observations",
    "Dev_Customers",
    "PR_AUC",
    "Precision_Top_1pct",
    "Precision_Top_2pct",
    "Precision_Top_5pct",
    "Precision_Top_10pct"
]

print(
    window_comparison[
        comparison_columns
    ].round(5).to_string(index=False)
)

window_comparison.to_csv(
    output_folder /
    "rfm_window_comparison.csv",
    index=False
)

print("\nRFM window experiment completed!")


# ==========================================
# FINAL EXPERIMENT
# HISTORICAL RFM + RECENT ACTIVITY
# ==========================================

print("\n========== FINAL FEATURE EXPERIMENT ==========")

# Use the same training and Dev snapshots
# from the previous experiment.

FINAL_TRAIN_SNAPSHOTS = RFM_TRAIN_SNAPSHOTS
FINAL_DEV_SNAPSHOT = RFM_DEV_SNAPSHOT

# Historical RFM features
HISTORY_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "customer_tenure",
    "avg_days_between_orders"
]

# Recent purchase activity
RECENT_FEATURES = [
    "orders_last_30d",
    "orders_last_90d"
]

COMBINED_FEATURES = (
    HISTORY_FEATURES + RECENT_FEATURES
)


# ==========================================
# 1. BUILD COMBINED SNAPSHOT
# ==========================================

def build_combined_snapshot(snapshot_date):

    # Existing feature engineering function
    snapshot = build_customer_snapshot(
        prediction_orders,
        snapshot_date
    )

    return snapshot


# ==========================================
# 2. PREPARE TRAINING DATA
# ==========================================

combined_training_parts = []

for snapshot_date in FINAL_TRAIN_SNAPSHOTS:

    label_end = (
        snapshot_date
        + pd.Timedelta(days=HORIZON_DAYS)
    )

    if label_end > FINAL_DEV_SNAPSHOT:
        raise ValueError(
            "Training labels overlap Dev."
        )

    combined_training_parts.append(
        build_combined_snapshot(snapshot_date)
    )

combined_train = pd.concat(
    combined_training_parts,
    ignore_index=True
)

combined_dev = build_combined_snapshot(
    FINAL_DEV_SNAPSHOT
)


# ==========================================
# 3. DEFINE FEATURE EXPERIMENTS
# ==========================================

FINAL_EXPERIMENTS = {
    "Historical RFM": HISTORY_FEATURES,

    "Historical RFM + Recent Activity":
        COMBINED_FEATURES
}


# ==========================================
# 4. TRAIN AND EVALUATE
# ==========================================

final_results = []

for experiment_name, feature_columns in FINAL_EXPERIMENTS.items():

    print(
        f"\n========== {experiment_name} =========="
    )

    X_train_final = combined_train[
        feature_columns
    ].copy()

    y_train_final = combined_train[
        "will_purchase_again"
    ].copy()

    X_dev_final = combined_dev[
        feature_columns
    ].copy()

    y_dev_final = combined_dev[
        "will_purchase_again"
    ].copy()

    # Validate missing values
    if X_train_final.isna().any().any():
        raise ValueError(
            "Missing training features."
        )

    if X_dev_final.isna().any().any():
        raise ValueError(
            "Missing Dev features."
        )

    # Same model for both experiments
    model = clone(base_model)

    model.fit(
        X_train_final,
        y_train_final
    )

    probabilities = model.predict_proba(
        X_dev_final
    )[:, 1]

    # PR-AUC
    pr_auc = average_precision_score(
        y_dev_final,
        probabilities
    )

    # Precision at Top K%
    topk = precision_at_top_percent(
        y_dev_final,
        probabilities,
        RFM_TOP_K
    )

    result = {
        "Experiment": experiment_name,
        "Train_Observations": len(combined_train),
        "Dev_Customers": len(combined_dev),
        "PR_AUC": pr_auc,
        **topk
    }

    final_results.append(result)

    print("PR-AUC:", round(pr_auc, 5))

    for percentage in RFM_TOP_K:

        print(
            f"Top {percentage}%:",
            "Precision =",
            round(
                topk[
                    f"Precision_Top_{percentage}pct"
                ],
                5
            ),
            "| Returning =",
            topk[
                f"Returning_Top_{percentage}pct"
            ]
        )


# ==========================================
# 5. FINAL COMPARISON
# ==========================================

final_comparison = pd.DataFrame(
    final_results
)

# Add the previous 90-day RFM result
# for comparison.

previous_90 = window_comparison[
    window_comparison["Window"] == "90 Days"
].copy()

if len(previous_90) == 1:

    previous_90["Experiment"] = "90-Day RFM"

    final_comparison = pd.concat(
        [
            final_comparison,
            previous_90[
                final_comparison.columns
            ]
        ],
        ignore_index=True
    )

final_comparison = final_comparison.sort_values(
    "PR_AUC",
    ascending=False
)

print(
    "\n========== FINAL FEATURE COMPARISON =========="
)

print(
    final_comparison[
        [
            "Experiment",
            "Train_Observations",
            "Dev_Customers",
            "PR_AUC",
            "Precision_Top_1pct",
            "Precision_Top_2pct",
            "Precision_Top_5pct",
            "Precision_Top_10pct"
        ]
    ].round(5).to_string(index=False)
)

final_comparison.to_csv(
    output_folder /
    "final_feature_comparison.csv",
    index=False
)

print("\nFinal feature experiment completed!")
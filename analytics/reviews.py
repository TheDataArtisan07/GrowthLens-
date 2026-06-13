import pandas as pd
import numpy as np

def calculate_average_review_score(df: pd.DataFrame) -> float:
    """
    Returns the average review score, dropping missing values.
    """
    if df is None or df.empty or 'review_score' not in df.columns:
        return 0.0
    return float(df['review_score'].dropna().mean())

def review_score_distribution(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary of review score distribution (1 to 5).
    """
    if df is None or df.empty or 'review_score' not in df.columns:
        return {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    counts = df['review_score'].dropna().value_counts().to_dict()
    # Ensure all keys from 1 to 5 are present
    return {i: int(counts.get(i, 0)) for i in range(1, 6)}

def reviews_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame containing average review score and review count
    grouped by product category name (English).
    """
    if df is None or df.empty or 'review_score' not in df.columns or 'product_category_name_english' not in df.columns:
        return pd.DataFrame(columns=['product_category_name_english', 'avg_review_score', 'review_count'])
    
    grouped = df.groupby('product_category_name_english').agg(
        avg_review_score=('review_score', 'mean'),
        review_count=('review_score', 'count')
    ).reset_index()
    return grouped.sort_values(by='review_count', ascending=False)

def reviews_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame containing average review score and review count
    grouped by customer state.
    """
    if df is None or df.empty or 'review_score' not in df.columns or 'customer_state' not in df.columns:
        return pd.DataFrame(columns=['customer_state', 'avg_review_score', 'review_count'])
    
    grouped = df.groupby('customer_state').agg(
        avg_review_score=('review_score', 'mean'),
        review_count=('review_score', 'count')
    ).reset_index()
    return grouped.sort_values(by='avg_review_score', ascending=False)

def delivery_delay_analysis(df: pd.DataFrame) -> dict:
    """
    Analyzes delivery delay by comparing actual delivery date with estimated delivery date.
    Calculates Late Delivery Rate, On-Time Delivery Rate, and Average Delay Days.
    """
    result = {
        "late_delivery_rate": 0.0,
        "on_time_delivery_rate": 0.0,
        "avg_delay_days": 0.0
    }
    if df is None or df.empty:
        return result
        
    if 'order_delivered_customer_date' not in df.columns or 'order_estimated_delivery_date' not in df.columns:
        return result
        
    # Filter rows that have both delivered and estimated dates (delivered orders)
    order_df = df.drop_duplicates(subset=['order_id']).dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date'])
    
    if order_df.empty:
        return result
        
    # Compute late delivery flag
    late_flags = order_df['order_delivered_customer_date'] > order_df['order_estimated_delivery_date']
    
    total_delivered = len(order_df)
    late_count = late_flags.sum()
    
    late_rate = (late_count / total_delivered) * 100
    on_time_rate = 100.0 - late_rate
    
    # Calculate delay days for late orders only (in float days)
    late_orders = order_df[late_flags]
    if not late_orders.empty:
        delay_deltas = late_orders['order_delivered_customer_date'] - late_orders['order_estimated_delivery_date']
        avg_delay = float(delay_deltas.dt.total_seconds().mean() / 86400.0)
    else:
        avg_delay = 0.0
        
    return {
        "late_delivery_rate": float(late_rate),
        "on_time_delivery_rate": float(on_time_rate),
        "avg_delay_days": float(avg_delay)
    }

def delivery_vs_review_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a clean DataFrame containing review_score and delivery_days for box plot analysis.
    """
    if df is None or df.empty or 'review_score' not in df.columns or 'delivery_days' not in df.columns:
        return pd.DataFrame(columns=['review_score', 'delivery_days'])
        
    clean_df = df.drop_duplicates(subset=['order_id']).dropna(subset=['review_score', 'delivery_days'])
    # Filter outliers to keep the box plot visualization clean
    clean_df = clean_df[(clean_df['delivery_days'] >= 0) & (clean_df['delivery_days'] <= 90)]
    return clean_df[['review_score', 'delivery_days']].sort_values(by='review_score')

def low_rating_categories(df: pd.DataFrame, min_reviews: int = 10) -> pd.DataFrame:
    """
    Finds product categories with average review score below threshold,
    filtering for categories with a minimum volume of reviews to avoid single-item noise.
    """
    if df is None or df.empty or 'review_score' not in df.columns or 'product_category_name_english' not in df.columns:
        return pd.DataFrame(columns=['product_category_name_english', 'avg_review_score', 'review_count'])
        
    grouped = df.groupby('product_category_name_english').agg(
        avg_review_score=('review_score', 'mean'),
        review_count=('review_score', 'count')
    ).reset_index()
    
    filtered = grouped[grouped['review_count'] >= min_reviews]
    return filtered.sort_values(by='avg_review_score', ascending=True)

def satisfaction_by_retention(df: pd.DataFrame) -> dict:
    """
    Measures and compares satisfaction levels (average review scores)
    between repeat customers (>1 unique orders) and one-time customers (exactly 1 unique order).
    """
    result = {
        "repeat_avg_rating": 0.0,
        "one_time_avg_rating": 0.0,
        "difference": 0.0
    }
    if df is None or df.empty or 'customer_unique_id' not in df.columns or 'review_score' not in df.columns:
        return result
        
    # Group by customer_unique_id to find unique order count and average rating
    customer_orders = df.groupby('customer_unique_id').agg(
        order_count=('order_id', 'nunique'),
        avg_customer_rating=('review_score', 'mean')
    ).reset_index()
    
    repeat_custs = customer_orders[customer_orders['order_count'] > 1]
    one_time_custs = customer_orders[customer_orders['order_count'] == 1]
    
    repeat_avg = float(repeat_custs['avg_customer_rating'].dropna().mean()) if not repeat_custs.empty else 0.0
    one_time_avg = float(one_time_custs['avg_customer_rating'].dropna().mean()) if not one_time_custs.empty else 0.0
    
    return {
        "repeat_avg_rating": repeat_avg,
        "one_time_avg_rating": one_time_avg,
        "difference": repeat_avg - one_time_avg
    }

def customer_satisfaction_metrics(df: pd.DataFrame) -> dict:
    """
    Calculates master customer satisfaction KPIs and aggregates.
    """
    avg_score = calculate_average_review_score(df)
    dist = review_score_distribution(df)
    total_reviews = sum(dist.values())
    
    pos_count = dist.get(4, 0) + dist.get(5, 0)
    neg_count = dist.get(1, 0) + dist.get(2, 0)
    neu_count = dist.get(3, 0)
    
    pos_rate = (pos_count / total_reviews * 100) if total_reviews > 0 else 0.0
    neg_rate = (neg_count / total_reviews * 100) if total_reviews > 0 else 0.0
    neu_rate = (neu_count / total_reviews * 100) if total_reviews > 0 else 0.0
    
    # Delivery delay stats
    delay_stats = delivery_delay_analysis(df)
    
    # Average delivery days
    avg_delivery_days = 0.0
    if df is not None and not df.empty and 'delivery_days' in df.columns:
        avg_delivery_days = float(df.drop_duplicates(subset=['order_id'])['delivery_days'].dropna().mean())
        
    # Lowest rated category
    low_cats = low_rating_categories(df, min_reviews=10)
    lowest_category = "N/A"
    lowest_rating = 0.0
    if not low_cats.empty:
        lowest_category = low_cats.iloc[0]['product_category_name_english']
        lowest_rating = float(low_cats.iloc[0]['avg_review_score'])
        
    return {
        "avg_review_score": avg_score,
        "positive_rate": pos_rate,
        "negative_rate": neg_rate,
        "neutral_rate": neu_rate,
        "avg_delivery_days": avg_delivery_days,
        "late_delivery_rate": delay_stats["late_delivery_rate"],
        "on_time_delivery_rate": delay_stats["on_time_delivery_rate"],
        "avg_delay_days": delay_stats["avg_delay_days"],
        "lowest_category": lowest_category,
        "lowest_category_rating": lowest_rating
    }

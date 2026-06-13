import pandas as pd
import numpy as np

def calculate_total_customers(df: pd.DataFrame) -> int:
    """
    Returns the count of unique customers.
    """
    if df is None or df.empty:
        return 0
    return int(df['customer_unique_id'].dropna().nunique())

def calculate_repeat_customers(df: pd.DataFrame) -> int:
    """
    Returns the count of customers with more than 1 order.
    """
    if df is None or df.empty:
        return 0
    order_counts = df.groupby('customer_unique_id')['order_id'].nunique()
    return int((order_counts > 1).sum())

def calculate_one_time_customers(df: pd.DataFrame) -> int:
    """
    Returns the count of customers with exactly 1 order.
    """
    if df is None or df.empty:
        return 0
    order_counts = df.groupby('customer_unique_id')['order_id'].nunique()
    return int((order_counts == 1).sum())

def customer_distribution_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts unique customers per state.
    Returns DataFrame with columns ['customer_state', 'customer_count'] sorted descending.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['customer_state', 'customer_count'])
    
    unique_custs = df.drop_duplicates(subset=['customer_unique_id'])
    state_dist = unique_custs.groupby('customer_state')['customer_unique_id'].count().reset_index()
    state_dist.columns = ['customer_state', 'customer_count']
    return state_dist.sort_values(by='customer_count', ascending=False)

def customer_purchase_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the distribution of number of orders placed by customers.
    Returns DataFrame with columns ['order_count', 'customer_count'] sorted by order count.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['order_count', 'customer_count'])
    
    order_counts = df.groupby('customer_unique_id')['order_id'].nunique().reset_index()
    order_counts.columns = ['customer_unique_id', 'order_count']
    
    freq_df = order_counts.groupby('order_count')['customer_unique_id'].count().reset_index()
    freq_df.columns = ['order_count', 'customer_count']
    return freq_df.sort_values(by='order_count')

def top_customers_by_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates top 20 customers by total payment value.
    Returns DataFrame with columns ['customer_unique_id', 'payment_value'] sorted descending.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['customer_unique_id', 'payment_value'])
    
    # We group by unique customer and sum their unique payments
    order_payments = df.drop_duplicates(subset=['order_id', 'payment_value'])
    top_custs = order_payments.groupby('customer_unique_id')['payment_value'].sum().reset_index()
    return top_custs.sort_values(by='payment_value', ascending=False).head(20)

def customer_lifetime_metrics(df: pd.DataFrame) -> float:
    """
    Returns the average spending (lifetime value) per unique customer.
    """
    if df is None or df.empty:
        return 0.0
    
    # Total revenue / unique customers
    order_payments = df.drop_duplicates(subset=['order_id', 'payment_value'])
    total_rev = order_payments['payment_value'].sum()
    unique_custs = df['customer_unique_id'].nunique()
    
    if unique_custs == 0:
        return 0.0
    return float(total_rev / unique_custs)

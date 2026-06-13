import pandas as pd
import numpy as np

def calculate_total_revenue(df: pd.DataFrame) -> float:
    """
    Returns the sum of distinct order payments.
    """
    if df is None or df.empty:
        return 0.0
    # Drop duplicate order payment entries to get unique payments
    order_payments = df.drop_duplicates(subset=['order_id', 'payment_value'])
    return float(order_payments['payment_value'].sum())

def calculate_average_order_value(df: pd.DataFrame) -> float:
    """
    Returns the average order payment value.
    """
    if df is None or df.empty:
        return 0.0
    total_rev = calculate_total_revenue(df)
    total_ords = calculate_total_orders(df)
    if total_ords == 0:
        return 0.0
    return float(total_rev / total_ords)

def calculate_total_orders(df: pd.DataFrame) -> int:
    """
    Returns the count of unique order IDs.
    """
    if df is None or df.empty:
        return 0
    return int(df['order_id'].dropna().nunique())

def monthly_revenue_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups unique order payments by purchase_year_month.
    Returns DataFrame with columns ['purchase_year_month', 'payment_value'] sorted chronologically.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['purchase_year_month', 'payment_value'])
    # Avoid duplicate payment summation for multi-item orders
    order_payments = df.drop_duplicates(subset=['order_id', 'payment_value'])
    monthly = order_payments.groupby('purchase_year_month')['payment_value'].sum().reset_index()
    return monthly.sort_values('purchase_year_month')

def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups item prices (order_value) by English category names.
    Returns DataFrame with columns ['product_category_name_english', 'order_value'] sorted descending.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['product_category_name_english', 'order_value'])
    
    # We aggregate on item-level order_value (price)
    category_df = df.groupby('product_category_name_english')['order_value'].sum().reset_index()
    return category_df.sort_values(by='order_value', ascending=False)

def revenue_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups order payments by customer state.
    Returns DataFrame with columns ['customer_state', 'payment_value'] sorted descending.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['customer_state', 'payment_value'])
    # Group unique payments by state
    order_payments = df.drop_duplicates(subset=['order_id', 'payment_value'])
    state_df = order_payments.groupby('customer_state')['payment_value'].sum().reset_index()
    return state_df.sort_values(by='payment_value', ascending=False)

def top_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates top categories by order_value (price) as a readable proxy for top products.
    Returns DataFrame with columns ['product_category_name_english', 'order_value'] sorted descending.
    """
    # For Olist, English product category functions as the descriptive display label
    return revenue_by_category(df)

def monthly_growth_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely computes MoM growth percentage checking for zero-revenue months.
    Returns DataFrame with columns ['purchase_year_month', 'payment_value', 'growth_rate'].
    """
    monthly_rev = monthly_revenue_trend(df)
    if monthly_rev.empty:
        return monthly_rev
        
    growth_rates = []
    for i in range(len(monthly_rev)):
        if i == 0:
            growth_rates.append(0.0)  # Safe default for first month
        else:
            prev_val = monthly_rev.iloc[i-1]['payment_value']
            curr_val = monthly_rev.iloc[i]['payment_value']
            if prev_val == 0 or pd.isna(prev_val):
                growth_rates.append(0.0)  # Safe division-by-zero check
            else:
                rate = ((curr_val - prev_val) / prev_val) * 100
                growth_rates.append(rate)
                
    monthly_rev['growth_rate'] = growth_rates
    return monthly_rev

def monthly_aov_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups order payments by purchase_year_month to calculate average order value trend.
    Returns DataFrame with columns ['purchase_year_month', 'aov'].
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['purchase_year_month', 'aov'])
    order_payments = df.drop_duplicates(subset=['order_id', 'payment_value'])
    
    monthly_stats = order_payments.groupby('purchase_year_month').agg(
        total_value=('payment_value', 'sum'),
        total_orders=('order_id', 'nunique')
    ).reset_index()
    
    monthly_stats['aov'] = monthly_stats['total_value'] / monthly_stats['total_orders']
    return monthly_stats[['purchase_year_month', 'aov']]

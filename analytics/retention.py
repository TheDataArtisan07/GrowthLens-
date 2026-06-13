import pandas as pd
import numpy as np

def calculate_repeat_purchase_rate(df: pd.DataFrame) -> float:
    """
    Returns the repeat purchase rate: unique buyers with >1 orders divided by total unique buyers.
    """
    if df is None or df.empty:
        return 0.0
    order_counts = df.groupby('customer_unique_id')['order_id'].nunique()
    total_custs = len(order_counts)
    if total_custs == 0:
        return 0.0
    repeat_custs = (order_counts > 1).sum()
    return float((repeat_custs / total_custs) * 100)

def calculate_retention_rate_m1(matrix: pd.DataFrame) -> float:
    """
    Returns the average Month 1 retention rate across all cohorts in the matrix.
    """
    if matrix is None or matrix.empty or 1 not in matrix.columns:
        return 0.0
    # Average of Month 1 (excluding NaN values which represent cohorts with no Month 1 yet)
    return float(matrix[1].dropna().mean())

def calculate_business_churn_rate(rfm: pd.DataFrame) -> float:
    """
    Returns the business churn rate (RFM-based): percentage of customers in Lost or At Risk segments.
    """
    if rfm is None or rfm.empty:
        return 0.0
    churn_custs = rfm['segment'].isin(['At Risk', 'Lost Customers']).sum()
    total_custs = len(rfm)
    if total_custs == 0:
        return 0.0
    return float((churn_custs / total_custs) * 100)

def calculate_retention_churn_rate(matrix: pd.DataFrame) -> float:
    """
    Returns the cohort-based retention churn rate: percentage of customers who never returned in Month 1.
    Formulated as: 100% - average Month 1 retention rate.
    """
    m1_rate = calculate_retention_rate_m1(matrix)
    return float(100.0 - m1_rate)

def calculate_average_customer_lifetime(lifespans: list) -> float:
    """
    Returns the average customer lifetime (span in days between first and latest purchase) across all users.
    """
    if not lifespans:
        return 0.0
    return float(np.mean(lifespans))

def calculate_customer_return_intervals(intervals: list) -> float:
    """
    Returns the average days between consecutive purchases for repeat buyers.
    """
    if not intervals:
        return 0.0
    return float(np.mean(intervals))

def find_best_performing_cohort(matrix: pd.DataFrame, sizes: dict, threshold: int = 1000) -> tuple:
    """
    Finds the cohort with the highest Month 1 retention rate, excluding cohorts with sample size < threshold.
    Returns (cohort_name, cohort_size, m1_retention_percentage).
    """
    if matrix is None or matrix.empty or 1 not in matrix.columns or not sizes:
        return "N/A", 0, 0.0
    
    best_cohort = "N/A"
    best_m1_rate = -1.0
    best_size = 0
    
    # Iterate through each cohort index in matrix
    for cohort_name in matrix.index:
        size = sizes.get(cohort_name, 0)
        # Check size threshold
        if size >= threshold:
            m1_val = matrix.loc[cohort_name, 1]
            if not pd.isna(m1_val) and m1_val > best_m1_rate:
                best_m1_rate = m1_val
                best_cohort = cohort_name
                best_size = size
                
    if best_cohort == "N/A":
        # Fallback if no cohort passes the size threshold
        for cohort_name in matrix.index:
            m1_val = matrix.loc[cohort_name, 1]
            if not pd.isna(m1_val) and m1_val > best_m1_rate:
                best_m1_rate = m1_val
                best_cohort = cohort_name
                best_size = sizes.get(cohort_name, 0)
                
    return best_cohort, int(best_size), float(max(0.0, best_m1_rate))

import pandas as pd
import numpy as np
import analytics.revenue as rev_calc
import analytics.customers as cust_calc
import analytics.retention as ret_calc
import analytics.reviews as rev_sat_calc

def add_months(month_str, num_months):
    """
    Utility to add months to a YYYY-MM string.
    """
    year = int(month_str.split('-')[0])
    month = int(month_str.split('-')[1])
    for _ in range(num_months):
        month += 1
        if month > 12:
            month = 1
            year += 1
    return f"{year:04d}-{month:02d}"

def calculate_r_squared(y_actual, y_fit):
    """
    Calculates the R-squared coefficient of determination.
    """
    y_mean = np.mean(y_actual)
    ss_tot = np.sum((y_actual - y_mean) ** 2)
    ss_res = np.sum((y_actual - y_fit) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

def get_revenue_forecast(df: pd.DataFrame) -> dict:
    """
    Predicts monthly revenue for the next 3 months using linear regression.
    Displays projected values, direction, and R-squared based confidence score (0-100%).
    """
    if df is None or df.empty:
        return {
            "history": [],
            "forecast": [],
            "slope": 0.0,
            "confidence_score": 0,
            "confidence_rating": "Low",
            "direction": "Stable"
        }
        
    monthly_rev = rev_calc.monthly_revenue_trend(df)
    
    # Filter low volume months (<100 orders)
    orders_per_month = df.groupby('purchase_year_month')['order_id'].nunique().loc[lambda x: x >= 100]
    monthly_rev = monthly_rev[monthly_rev['purchase_year_month'].isin(orders_per_month.index)].sort_values('purchase_year_month')
    
    if len(monthly_rev) < 2:
        return {
            "history": monthly_rev.to_dict(orient='records'),
            "forecast": [],
            "slope": 0.0,
            "confidence_score": 0,
            "confidence_rating": "Low",
            "direction": "Stable"
        }
        
    x = np.arange(len(monthly_rev))
    y = monthly_rev['payment_value'].values
    
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    r_squared = calculate_r_squared(y, y_fit)
    
    conf_score = int(max(0, min(100, r_squared * 100)))
    if conf_score >= 70:
        conf_rating = "High"
    elif conf_score >= 40:
        conf_rating = "Medium"
    else:
        conf_rating = "Low"
        
    # Extrapolate 3 months
    last_month_str = monthly_rev.iloc[-1]['purchase_year_month']
    forecast_points = []
    x_new = np.arange(len(monthly_rev), len(monthly_rev) + 3)
    
    for i, x_val in enumerate(x_new):
        month_label = add_months(last_month_str, i + 1)
        pred_val = max(0.0, slope * x_val + intercept)
        forecast_points.append({
            "purchase_year_month": month_label,
            "payment_value": float(pred_val),
            "is_forecast": True
        })
        
    history_points = monthly_rev.copy()
    history_points['is_forecast'] = False
    history_list = history_points.to_dict(orient='records')
    
    return {
        "history": history_list,
        "forecast": forecast_points,
        "slope": float(slope),
        "confidence_score": conf_score,
        "confidence_rating": conf_rating,
        "direction": "Upward" if slope > 0.05 else ("Downward" if slope < -0.05 else "Stable")
    }

def get_customer_growth_forecast(df: pd.DataFrame) -> dict:
    """
    Predicts future customer acquisitions (new) and repeat buyers (repeat) for the next 3 months.
    """
    default_res = {"history": [], "forecast": []}
    if df is None or df.empty:
        return default_res
        
    # First purchase per customer
    first_purchase = df.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
    first_purchase.columns = ['customer_unique_id', 'first_purchase_time']
    df_with_fp = df.merge(first_purchase, on='customer_unique_id', how='left')
    df_with_fp['first_purchase_year_month'] = df_with_fp['first_purchase_time'].dt.strftime('%Y-%m')
    
    # Months with substantial data
    orders_per_month = df.groupby('purchase_year_month')['order_id'].nunique().loc[lambda x: x >= 100]
    months_sorted = sorted(orders_per_month.index)
    
    history_points = []
    new_buyers_list = []
    rep_buyers_list = []
    
    for m in months_sorted:
        m_df = df_with_fp[df_with_fp['purchase_year_month'] == m]
        total_unique = m_df['customer_unique_id'].nunique()
        new_unique = m_df[m_df['first_purchase_year_month'] == m]['customer_unique_id'].nunique()
        rep_unique = total_unique - new_unique
        
        new_buyers_list.append(new_unique)
        rep_buyers_list.append(rep_unique)
        
        history_points.append({
            "purchase_year_month": m,
            "new_customers": int(new_unique),
            "repeat_customers": int(rep_unique),
            "is_forecast": False
        })
        
    if len(history_points) < 2:
        return {"history": history_points, "forecast": []}
        
    x = np.arange(len(history_points))
    
    # Fit trends
    new_slope, new_intercept = np.polyfit(x, new_buyers_list, 1)
    rep_slope, rep_intercept = np.polyfit(x, rep_buyers_list, 1)
    
    last_month_str = months_sorted[-1]
    forecast_points = []
    x_new = np.arange(len(history_points), len(history_points) + 3)
    
    for i, x_val in enumerate(x_new):
        month_label = add_months(last_month_str, i + 1)
        new_pred = max(0, int(new_slope * x_val + new_intercept))
        rep_pred = max(0, int(rep_slope * x_val + rep_intercept))
        
        forecast_points.append({
            "purchase_year_month": month_label,
            "new_customers": new_pred,
            "repeat_customers": rep_pred,
            "is_forecast": True
        })
        
    return {
        "history": history_points,
        "forecast": forecast_points,
        "new_slope": float(new_slope),
        "repeat_slope": float(rep_slope)
    }

def get_churn_risk_analysis(rfm_df: pd.DataFrame) -> dict:
    """
    Classifies customer churn risk using RFM segments:
    - Low Risk: Champions, Loyal Customers, Potential Loyalists
    - Medium Risk: Recent Customers, Need Attention
    - High Risk: At Risk, Lost Customers
    """
    if rfm_df is None or rfm_df.empty:
        return {"Low Risk": {"count": 0, "pct": 0.0}, "Medium Risk": {"count": 0, "pct": 0.0}, "High Risk": {"count": 0, "pct": 0.0}}
        
    total_customers = len(rfm_df)
    
    low_risk_segments = ['Champions', 'Loyal Customers', 'Potential Loyalists']
    med_risk_segments = ['Recent Customers', 'Need Attention']
    high_risk_segments = ['At Risk', 'Lost Customers']
    
    low_count = int(rfm_df['segment'].isin(low_risk_segments).sum())
    med_count = int(rfm_df['segment'].isin(med_risk_segments).sum())
    high_count = int(rfm_df['segment'].isin(high_risk_segments).sum())
    
    # Any residual segments map to medium risk as a catch-all
    other_count = total_customers - (low_count + med_count + high_count)
    med_count += other_count
    
    def pct(count):
        return float((count / total_customers) * 100) if total_customers > 0 else 0.0
        
    return {
        "Low Risk": {"count": low_count, "pct": pct(low_count)},
        "Medium Risk": {"count": med_count, "pct": pct(med_count)},
        "High Risk": {"count": high_count, "pct": pct(high_count)},
        "total": total_customers
    }

def get_category_outlook(df: pd.DataFrame) -> dict:
    """
    Evaluates category sales projections by fitting linear slopes over the last 6 months of data.
    Identifies the top 3 fastest growing and top 3 weakest categories.
    """
    if df is None or df.empty:
        return {"growing": [], "weakening": []}
        
    # Group category sales by month
    cat_monthly = df.groupby(['product_category_name_english', 'purchase_year_month'])['order_value'].sum().reset_index()
    
    # Get last 6 months list
    months_list = sorted(df['purchase_year_month'].unique())
    last_6_months = months_list[-6:] if len(months_list) >= 6 else months_list
    
    # Focus on top 15 categories by total revenue to avoid low-volume noise
    top_cats = df.groupby('product_category_name_english')['order_value'].sum().reset_index()
    top_cats = top_cats.sort_values(by='order_value', ascending=False).head(15)['product_category_name_english'].tolist()
    
    category_slopes = []
    
    for cat in top_cats:
        cat_df = cat_monthly[(cat_monthly['product_category_name_english'] == cat) & (cat_monthly['purchase_year_month'].isin(last_6_months))].sort_values('purchase_year_month')
        
        if len(cat_df) < 3:
            continue
            
        x = np.arange(len(cat_df))
        y = cat_df['order_value'].values
        
        slope, _ = np.polyfit(x, y, 1)
        
        category_slopes.append({
            "category": cat.replace('_', ' ').title(),
            "slope": float(slope),
            "recent_sales": float(y[-1]),
            "trend": "Expansion" if slope > 0 else "Contraction"
        })
        
    if not category_slopes:
        return {"growing": [], "weakening": []}
        
    # Sort by slope
    sorted_cats = sorted(category_slopes, key=lambda x: x['slope'], reverse=True)
    growing = sorted_cats[:3]
    weakening = sorted(sorted_cats, key=lambda x: x['slope'])[:3]
    
    return {
        "growing": growing,
        "weakening": weakening
    }

def generate_forecast_summary(df: pd.DataFrame, rfm_df: pd.DataFrame, revenue_forecast: dict, customer_forecast: dict) -> dict:
    """
    Generates summary cards: Key Forecast, Biggest Opportunity, Biggest Risk, Recommended Action.
    """
    rev_direction = revenue_forecast.get("direction", "Stable")
    rev_next_month = revenue_forecast["forecast"][0]["payment_value"] if revenue_forecast.get("forecast") else 0.0
    
    # Churn Risk numbers
    churn_analysis = get_churn_risk_analysis(rfm_df)
    high_risk_count = churn_analysis["High Risk"]["count"]
    
    cust_acq_slope = customer_forecast.get("new_slope", 0.0)
    
    key_forecast = f"Revenue is projected to follow a **{rev_direction}** trend over the next 3 months, with next-month sales estimated at **${rev_next_month:,.2f}**."
    
    biggest_opportunity = f"Reactivating the **{high_risk_count:,}** customers classified as High Churn Risk, representing substantial dormant value."
    
    if cust_acq_slope < 0:
        biggest_risk = f"New customer acquisition velocity is slowing down (**{cust_acq_slope:.1f} users/month**), threatening top-line growth."
    else:
        biggest_risk = "Dormant accounts accumulation outpaces repeat purchaser retention rates."
        
    recommended_action = "Deploy automated post-purchase drip incentives in month 1 and launch state-level shipping adjustments in high-delay regions."
    
    return {
        "key_forecast": key_forecast,
        "biggest_opportunity": biggest_opportunity,
        "biggest_risk": biggest_risk,
        "recommended_action": recommended_action
    }

def get_predictive_health_indicators(revenue_forecast: dict, customer_forecast: dict) -> dict:
    """
    Returns Predictive Health Indicators (Revenue, Customer, Retention, Overall Outlook).
    Values: Strong, Stable, Watch Closely, High Risk.
    """
    rev_slope = revenue_forecast.get("slope", 0.0)
    rev_conf = revenue_forecast.get("confidence_score", 0)
    
    new_slope = customer_forecast.get("new_slope", 0.0)
    rep_slope = customer_forecast.get("repeat_slope", 0.0)
    
    # 1. Revenue Outlook
    if rev_slope > 1000 and rev_conf >= 50:
        rev_outlook = {"status": "Strong", "badge": "success"}
    elif rev_slope >= -1000 and rev_slope <= 1000:
        rev_outlook = {"status": "Stable", "badge": "primary"}
    elif rev_slope < -1000 and rev_conf < 40:
        rev_outlook = {"status": "Watch Closely", "badge": "warning"}
    else:
        rev_outlook = {"status": "High Risk", "badge": "danger"}
        
    # 2. Customer Outlook
    if new_slope > 10:
        cust_outlook = {"status": "Strong", "badge": "success"}
    elif new_slope >= -10 and new_slope <= 10:
        cust_outlook = {"status": "Stable", "badge": "primary"}
    elif new_slope < -10 and new_slope > -50:
        cust_outlook = {"status": "Watch Closely", "badge": "warning"}
    else:
        cust_outlook = {"status": "High Risk", "badge": "danger"}
        
    # 3. Retention Outlook
    if rep_slope > 5:
        ret_outlook = {"status": "Strong", "badge": "success"}
    elif rep_slope >= -5 and rep_slope <= 5:
        ret_outlook = {"status": "Stable", "badge": "primary"}
    elif rep_slope < -5 and rep_slope > -20:
        ret_outlook = {"status": "Watch Closely", "badge": "warning"}
    else:
        ret_outlook = {"status": "High Risk", "badge": "danger"}
        
    # 4. Overall Outlook
    outlooks = [rev_outlook["status"], cust_outlook["status"], ret_outlook["status"]]
    if "High Risk" in outlooks:
        overall = {"status": "High Risk", "badge": "danger"}
    elif outlooks.count("Watch Closely") >= 2:
        overall = {"status": "Watch Closely", "badge": "warning"}
    elif outlooks.count("Strong") >= 2:
        overall = {"status": "Strong", "badge": "success"}
    else:
        overall = {"status": "Stable", "badge": "primary"}
        
    return {
        "revenue": rev_outlook,
        "customer": cust_outlook,
        "retention": ret_outlook,
        "overall": overall
    }

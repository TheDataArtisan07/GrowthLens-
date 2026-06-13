import pandas as pd
import numpy as np
import analytics.revenue as rev_calc
import analytics.customers as cust_calc
import analytics.retention as ret_calc
import analytics.reviews as rev_sat_calc

def get_action_center_snapshot(df: pd.DataFrame, retention_matrix: pd.DataFrame) -> dict:
    """
    Compiles the 6 snapshot cards: Revenue, Total Customers, Repeat Purchase Rate, 
    Retention Rate, Average Review Score, Top Product Category.
    """
    if df is None or df.empty:
        return {
            "revenue": 0.0,
            "customers": 0,
            "repeat_rate": 0.0,
            "retention_rate": 0.0,
            "avg_review_score": 0.0,
            "top_category": "N/A"
        }
        
    revenue = rev_calc.calculate_total_revenue(df)
    customers = cust_calc.calculate_total_customers(df)
    repeat_rate = ret_calc.calculate_repeat_purchase_rate(df)
    retention_rate = ret_calc.calculate_retention_rate_m1(retention_matrix)
    avg_review_score = rev_sat_calc.calculate_average_review_score(df)
    
    # Top Product Category
    top_cats = rev_calc.top_products(df)
    top_cat = "N/A"
    if not top_cats.empty:
        top_cat = top_cats.iloc[0]['product_category_name_english'].replace('_', ' ').title()
        
    return {
        "revenue": float(revenue),
        "customers": int(customers),
        "repeat_rate": float(repeat_rate),
        "retention_rate": float(retention_rate),
        "avg_review_score": float(avg_review_score),
        "top_category": top_cat
    }

def get_quick_metrics_trends(df: pd.DataFrame) -> dict:
    """
    Calculates Month-over-Month trend percentages and directions for:
    - Revenue Trend
    - Retention Trend (Active Repeat Share)
    - Satisfaction Trend
    - Customer Growth Trend
    """
    default_res = {
        "revenue": {"val": "0.0%", "arrow": "→", "class": "text-muted"},
        "retention": {"val": "0.0%", "arrow": "→", "class": "text-muted"},
        "satisfaction": {"val": "0.0%", "arrow": "→", "class": "text-muted"},
        "customers": {"val": "0.0%", "arrow": "→", "class": "text-muted"}
    }
    if df is None or df.empty:
        return default_res
        
    monthly_summary = df.groupby('purchase_year_month').agg(
        order_count=('order_id', 'nunique'),
        payment_sum=('payment_value', 'sum'),
        avg_review=('review_score', 'mean'),
        customer_count=('customer_unique_id', 'nunique')
    ).reset_index()
    
    valid_months = monthly_summary[monthly_summary['order_count'] >= 100].sort_values(by='purchase_year_month')
    if len(valid_months) < 2:
        return default_res
        
    month_curr = valid_months.iloc[-1]['purchase_year_month']
    month_prev = valid_months.iloc[-2]['purchase_year_month']
    
    # 1. Revenue
    rev_curr = float(valid_months.iloc[-1]['payment_sum'])
    rev_prev = float(valid_months.iloc[-2]['payment_sum'])
    rev_chg = ((rev_curr - rev_prev) / rev_prev * 100) if rev_prev > 0 else 0.0
    
    # 2. Active Repeat Share
    first_purchase = df.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
    first_purchase.columns = ['customer_unique_id', 'first_purchase_time']
    df_with_fp = df.merge(first_purchase, on='customer_unique_id', how='left')
    df_with_fp['is_repeat_purchase'] = df_with_fp['order_purchase_timestamp'] > df_with_fp['first_purchase_time']
    
    def get_repeat_share(month_str):
        month_df = df_with_fp[df_with_fp['purchase_year_month'] == month_str]
        if month_df.empty:
            return 0.0
        total_buyers = month_df['customer_unique_id'].nunique()
        repeat_buyers = month_df[month_df['is_repeat_purchase']]['customer_unique_id'].nunique()
        return (repeat_buyers / total_buyers * 100.0) if total_buyers > 0 else 0.0
        
    rep_curr = get_repeat_share(month_curr)
    rep_prev = get_repeat_share(month_prev)
    ret_chg = rep_curr - rep_prev
    
    # 3. Satisfaction
    sat_curr = float(valid_months.iloc[-1]['avg_review'])
    sat_prev = float(valid_months.iloc[-2]['avg_review'])
    sat_chg = ((sat_curr - sat_prev) / sat_prev * 100) if sat_prev > 0 else 0.0
    
    # 4. Customers
    cust_curr = float(valid_months.iloc[-1]['customer_count'])
    cust_prev = float(valid_months.iloc[-2]['customer_count'])
    cust_chg = ((cust_curr - cust_prev) / cust_prev * 100) if cust_prev > 0 else 0.0
    
    # Formatter
    def format_trend(change_val, is_pct_diff=False):
        arrow = "↑" if change_val > 0.05 else ("↓" if change_val < -0.05 else "→")
        color_class = "text-success" if change_val > 0.05 else ("text-danger" if change_val < -0.05 else "text-muted")
        symbol = "%" if not is_pct_diff else " pp" # percentage points for share diffs
        val_str = f"{arrow} {abs(change_val):.1f}{symbol}"
        return {"val": val_str, "arrow": arrow, "class": color_class}
        
    return {
        "revenue": format_trend(rev_chg),
        "retention": format_trend(ret_chg, is_pct_diff=True),
        "satisfaction": format_trend(sat_chg),
        "customers": format_trend(cust_chg)
    }

def identify_top_findings(df: pd.DataFrame, rfm_df: pd.DataFrame, retention_matrix: pd.DataFrame, reviews_metrics: dict) -> list:
    """
    Identifies 5 top findings dynamically based on calculation models.
    """
    findings = []
    if df is None or df.empty:
        return findings
        
    # Finding 1: MoM Revenue Growth
    trends = get_quick_metrics_trends(df)
    rev_val = trends["revenue"]["val"]
    is_up = trends["revenue"]["arrow"] == "↑"
    findings.append({
        "title": "MoM Revenue Trend",
        "detail": f"Monthly revenue { 'increased' if is_up else 'decreased' } by {rev_val.split(' ')[1]} based on last two months comparisons.",
        "icon": "bi-graph-up-arrow" if is_up else "bi-graph-down-arrow",
        "badge": "success" if is_up else "danger"
    })
    
    # Finding 2: One-time Buyers Concentration
    repeat_rate = ret_calc.calculate_repeat_purchase_rate(df)
    one_time_pct = 100.0 - repeat_rate
    findings.append({
        "title": "One-Time Purchases",
        "detail": f"{one_time_pct:.1f}% of our customer base purchase only once, posing a challenge to LTV expansion.",
        "icon": "bi-person-x",
        "badge": "warning"
    })
    
    # Finding 3: Customer Satisfaction Level
    avg_review = reviews_metrics.get("avg_review_score", 0.0)
    findings.append({
        "title": "Average Customer Rating",
        "detail": f"Average review score remains healthy at {avg_review:.2f} out of 5.0, reflecting strong overall satisfaction.",
        "icon": "bi-star-fill",
        "badge": "success"
    })
    
    # Finding 4: Cohort Retention Level
    m1_retention = ret_calc.calculate_retention_rate_m1(retention_matrix)
    findings.append({
        "title": "Month 1 Retention Rate",
        "detail": f"Average Month 1 customer cohort retention is below 5%, currently measured at {m1_retention:.2f}%.",
        "icon": "bi-funnel",
        "badge": "danger"
    })
    
    # Finding 5: Top Category Contribution
    top_cats = rev_calc.top_products(df)
    if not top_cats.empty:
        top_name = top_cats.iloc[0]['product_category_name_english'].replace('_', ' ').title()
        top_val = top_cats.iloc[0]['order_value']
        total_payment = df.drop_duplicates(subset=['order_id', 'payment_value'])['payment_value'].sum()
        contrib_pct = (top_val / total_payment * 100) if total_payment > 0 else 0.0
        findings.append({
            "title": "Top Category Concentration",
            "detail": f"Our top product category '{top_name}' contributes {contrib_pct:.1f}% of overall total platform revenue.",
            "icon": "bi-tag-fill",
            "badge": "info"
        })
        
    return findings[:5]

def get_working_well(df: pd.DataFrame, rfm_df: pd.DataFrame, retention_matrix: pd.DataFrame, reviews_metrics: dict) -> list:
    """
    Returns positive operational performance successes.
    """
    working_well = []
    if df is None or df.empty:
        return working_well
        
    # 1. Satisfaction
    pos_rate = reviews_metrics.get("positive_rate", 0.0)
    working_well.append({
        "title": "High Customer Satisfaction",
        "desc": f"Solid customer feedback sentiment with a positive review rate of {pos_rate:.1f}% (4-5 star reviews)."
    })
    
    # 2. Top categories
    top_cats = rev_calc.top_products(df)
    if not top_cats.empty:
        top_name = top_cats.iloc[0]['product_category_name_english'].replace('_', ' ').title()
        top_val = top_cats.iloc[0]['order_value']
        working_well.append({
            "title": f"Strong Category Leader: {top_name}",
            "desc": f"Top category '{top_name}' demonstrates robust sales volume, generating ${top_val:,.2f} in sales."
        })
        
    # 3. On-Time Delivery Rate
    on_time = reviews_metrics.get("on_time_delivery_rate", 0.0)
    working_well.append({
        "title": "Courier SLA Performance",
        "desc": f"Overall delivery logistics remain stable with an on-time shipment rate of {on_time:.1f}%."
    })
    
    # 4. Total Active Customers
    total_custs = cust_calc.calculate_total_customers(df)
    working_well.append({
        "title": "Growing Customer Base",
        "desc": f"Platform has established a broad customer footprint with {total_custs:,} unique buyers."
    })
    
    return working_well

def get_needs_attention(df: pd.DataFrame, rfm_df: pd.DataFrame, retention_matrix: pd.DataFrame, reviews_metrics: dict) -> list:
    """
    Returns negative business issues or operational constraints.
    """
    needs_attention = []
    if df is None or df.empty:
        return needs_attention
        
    # 1. Month 1 Cohort Retention
    m1_ret = ret_calc.calculate_retention_rate_m1(retention_matrix)
    needs_attention.append({
        "title": "Post-Purchase Churn Cliff",
        "desc": f"Severe cohort drop-off with average Month 1 retention resting at only {m1_ret:.2f}%."
    })
    
    # 2. One-Time Buyers
    rep_rate = ret_calc.calculate_repeat_purchase_rate(df)
    one_time_pct = 100.0 - rep_rate
    needs_attention.append({
        "title": "One-Time Buyer Dependency",
        "desc": f"Extremely high transactional single-purchase base: {one_time_pct:.1f}% of customers do not make a second order."
    })
    
    # 3. Delivery delays
    late_rate = reviews_metrics.get("late_delivery_rate", 0.0)
    avg_delay = reviews_metrics.get("avg_delay_days", 0.0)
    needs_attention.append({
        "title": "Late Delivery Courier Penalties",
        "desc": f"{late_rate:.1f}% of orders suffer shipment delays, averaging {avg_delay:.1f} days late and severely impacting review ratings."
    })
    
    # 4. Low rated categories
    low_cats = rev_sat_calc.low_rating_categories(df, min_reviews=10)
    if not low_cats.empty:
        worst_name = low_cats.iloc[0]['product_category_name_english'].replace('_', ' ').title()
        worst_rating = low_cats.iloc[0]['avg_review_score']
        needs_attention.append({
            "title": f"Low-Rated Category: {worst_name}",
            "desc": f"Quality or merchant issues in '{worst_name}' dragging average review score down to {worst_rating:.2f} stars."
        })
        
    return needs_attention

def get_recommended_actions(df: pd.DataFrame, rfm_df: pd.DataFrame, retention_matrix: pd.DataFrame, reviews_metrics: dict) -> list:
    """
    Generates structured recommended actions including Action, Why This Matters, Expected Impact, and Priority.
    """
    actions = []
    if df is None or df.empty:
        return actions
        
    rep_rate = ret_calc.calculate_repeat_purchase_rate(df)
    one_time_pct = 100.0 - rep_rate
    m1_ret = ret_calc.calculate_retention_rate_m1(retention_matrix)
    late_rate = reviews_metrics.get("late_delivery_rate", 0.0)
    
    # Action 1: Post-Purchase Email Retention Flow
    actions.append({
        "action": "Deploy Post-Purchase Retention Campaigns",
        "why_matters": f"{100.0 - m1_ret:.1f}% of customers churn immediately after Month 0.",
        "impact": "Double Cohort Month 1 Retention & increase lifetime customer equity.",
        "priority": "Critical"
    })
    
    # Action 2: Loyalty program
    actions.append({
        "action": "Launch Points-Based Customer Loyalty Program",
        "why_matters": f"{one_time_pct:.1f}% of our customers are one-time buyers.",
        "impact": "Boost repeat purchase frequency and lower customer acquisition costs.",
        "priority": "High"
    })
    
    # Action 3: Courier audits
    actions.append({
        "action": "Audit Logistics & Penalize Delayed Couriers",
        "why_matters": f"{late_rate:.1f}% late delivery rate drops average reviews from 4.30 to 2.27 stars.",
        "impact": "Recover positive satisfaction scores and reduce support tickets.",
        "priority": "High"
    })
    
    # Action 4: Promote high-performing categories
    top_cats = rev_calc.top_products(df)
    top_name = top_cats.iloc[0]['product_category_name_english'].replace('_', ' ').title() if not top_cats.empty else "Top Category"
    actions.append({
        "action": f"Expand Marketing Budgets for '{top_name}'",
        "why_matters": f"Top category '{top_name}' drives key platform revenue contribution.",
        "impact": "Accelerate top-line revenue growth with high-converting categories.",
        "priority": "Medium"
    })
    
    # Action 5: Upsell Loyal segment
    actions.append({
        "action": "Target Loyal Customers with Upselling Campaigns",
        "why_matters": "Loyal customers represent high active value but low segment count.",
        "impact": "Generate high-margin repeat sales with zero customer acquisition cost.",
        "priority": "Medium"
    })
    
    return actions

def get_top_opportunities(df: pd.DataFrame, rfm_df: pd.DataFrame, retention_matrix: pd.DataFrame) -> list:
    """
    Generates growth opportunities table with Opportunity, Reason, and Potential Benefit.
    """
    opportunities = []
    if df is None or df.empty or rfm_df is None or rfm_df.empty:
        return opportunities
        
    rep_rate = ret_calc.calculate_repeat_purchase_rate(df)
    m1_ret = ret_calc.calculate_retention_rate_m1(retention_matrix)
    
    # Opportunity 1: Retention Boost
    one_time_custs = rfm_df[rfm_df['frequency'] == 1]
    n_one_time = len(one_time_custs)
    pot_cust_ret = int(n_one_time * 0.05)
    pot_rev_ret = pot_cust_ret * 160.0 # AoV proxy $160
    opportunities.append({
        "opportunity": "Convert One-Time Buyers into Repeat Buyers",
        "reason": f"Month 1 retention rates are below 1.5% ({m1_ret:.2f}%), indicating vast room for follow-up orders.",
        "benefit": f"Improving Month 1 retention by 5.0% retains {pot_cust_ret:,} customers, generating ~${pot_rev_ret:,.2f} in repeat sales."
    })
    
    # Opportunity 2: Category Scaling
    top_cats = rev_calc.top_products(df)
    if not top_cats.empty:
        top_name = top_cats.iloc[0]['product_category_name_english'].replace('_', ' ').title()
        top_val = top_cats.iloc[0]['order_value']
        pot_rev_cat = 0.15 * top_val
        opportunities.append({
            "opportunity": f"Expand Sales inside '{top_name}' Category",
            "reason": f"'{top_name}' is our largest and highest-converting category engine on the platform.",
            "benefit": f"Scaling category sales by 15.0% will generate an estimated ~${pot_rev_cat:,.2f} in additional revenue."
        })
        
    # Opportunity 3: Logistics Improvement
    opportunities.append({
        "opportunity": "Improve Logistics in Delayed States (e.g. RJ & SP)",
        "reason": "Regional courier bottlenecks are leading to severe delivery delays and negative customer reviews.",
        "benefit": "Eliminating regional delivery delays can lift overall customer satisfaction average rating by ~0.15 stars."
    })
    
    return opportunities

def generate_executive_summary(df: pd.DataFrame, rfm_df: pd.DataFrame, retention_matrix: pd.DataFrame, reviews_metrics: dict) -> str:
    """
    Generates a concise 5-sentence executive summary.
    """
    total_rev = rev_calc.calculate_total_revenue(df)
    avg_rating = reviews_metrics.get("avg_review_score", 0.0)
    rep_rate = ret_calc.calculate_repeat_purchase_rate(df)
    one_time_pct = 100.0 - rep_rate
    m1_ret = ret_calc.calculate_retention_rate_m1(retention_matrix)
    
    s1 = f"Platform revenue remains healthy at ${total_rev/1000000:.2f}M, supported by a solid average review rating of {avg_rating:.2f} stars."
    s2 = f"However, customer retention remains our primary operational bottleneck, as {one_time_pct:.1f}% of customers purchase only once."
    s3 = f"Average Month 1 cohort retention stands at just {m1_ret:.2f}%, indicating a critical drop-off in post-purchase engagement."
    s4 = "The single highest-leverage growth opportunity lies in converting one-time buyers into loyal repeat purchasers."
    s5 = "We recommend prioritizing post-purchase retention email flows, State-level logistics audits, and customer loyalty initiatives."
    
    return " ".join([s1, s2, s3, s4, s5])

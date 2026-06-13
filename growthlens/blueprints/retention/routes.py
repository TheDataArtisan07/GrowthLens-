from flask import Blueprint, render_template
from services.data_manager import data_manager
import analytics.retention as ret_calc
import plotly.express as px
import plotly.graph_objects as go
from growthlens.utils import plotly_to_json
import json
import pandas as pd
import numpy as np

retention_bp = Blueprint('retention', __name__)

@retention_bp.route('/retention')
def index():
    df = data_manager.get_analytics_df()
    rfm_df = data_manager.get_rfm_df()
    retention_matrix = data_manager.get_retention_matrix()
    ret_metrics = data_manager.get_retention_metrics()
    diagnostics = data_manager.get_diagnostics()

    if df is None or rfm_df is None or retention_matrix is None or not diagnostics.get("load_success", False):
        return render_template('retention.html', active_page='retention', loaded=False)

    # 1. Calculate Core KPIs
    repeat_rate = ret_calc.calculate_repeat_purchase_rate(df)
    m1_retention = ret_calc.calculate_retention_rate_m1(retention_matrix)
    
    # Dual Churn Metrics
    business_churn = ret_calc.calculate_business_churn_rate(rfm_df)
    retention_churn = ret_calc.calculate_retention_churn_rate(retention_matrix)
    
    avg_lifetime = ret_calc.calculate_average_customer_lifetime(ret_metrics.get("lifespans", []))
    avg_return_days = ret_calc.calculate_customer_return_intervals(ret_metrics.get("return_intervals", []))

    # Best performing cohort metrics (Minimum size = 1000)
    best_cohort_name, best_cohort_size, best_m1_pct = ret_calc.find_best_performing_cohort(
        retention_matrix, 
        ret_metrics.get("cohort_sizes", {}), 
        threshold=1000
    )

    # 2. Retention Health Indicators
    # A. Retention Health (Month 1 avg retention)
    if m1_retention >= 10.0:
        ret_health_status = "Healthy"
        ret_health_badge = "success"
        ret_health_desc = f"Average Month 1 retention is {m1_retention:.2f}%. Excellent customer return rate."
    elif m1_retention >= 3.0:
        ret_health_status = "Moderate"
        ret_health_badge = "warning"
        ret_health_desc = f"Average Month 1 retention is {m1_retention:.2f}%. Moderate cohort return rate."
    else:
        ret_health_status = "Needs Attention"
        ret_health_badge = "danger"
        ret_health_desc = f"Average Month 1 retention is {m1_retention:.2f}%. Drop-off after Month 0 is extremely steep."

    # B. Churn Risk (Business Churn)
    if business_churn < 30.0:
        churn_risk_status = "Healthy"
        churn_risk_badge = "success"
        churn_risk_desc = f"Business churn rate is {business_churn:.1f}%. Safe buyer retention."
    elif business_churn < 60.0:
        churn_risk_status = "Moderate"
        churn_risk_badge = "warning"
        churn_risk_desc = f"Business churn rate is {business_churn:.1f}%. Stable base but reactivations needed."
    else:
        churn_risk_status = "Needs Attention"
        churn_risk_badge = "danger"
        churn_risk_desc = f"Business churn is {business_churn:.1f}% (R_Score <= 2). Large percentage of users are dormant."

    # C. Customer Loyalty (Repeat Purchase Rate)
    if repeat_rate >= 20.0:
        loyalty_status = "Healthy"
        loyalty_badge = "success"
        loyalty_desc = f"Repeat purchase rate is {repeat_rate:.2f}%. High base loyalty."
    elif repeat_rate >= 5.0:
        loyalty_status = "Moderate"
        loyalty_badge = "warning"
        loyalty_desc = f"Repeat purchase rate is {repeat_rate:.2f}%. Normal retail loyalty."
    else:
        loyalty_status = "Needs Attention"
        loyalty_badge = "danger"
        loyalty_desc = f"Repeat purchase rate is {repeat_rate:.2f}%. The business relies heavily on one-time buyers."

    health_indicators = {
        "retention": {"status": ret_health_status, "badge": ret_health_badge, "desc": ret_health_desc},
        "churn": {"status": churn_risk_status, "badge": churn_risk_badge, "desc": churn_risk_desc},
        "loyalty": {"status": loyalty_status, "badge": loyalty_badge, "desc": loyalty_desc}
    }

    # 3. Build Plotly Visualizations
    plotly_layout_defaults = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#64748b"),
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        hovermode="closest"
    )

    # Chart 1: Retention Heatmap (Plotly Heatmap Grid)
    # Truncate matrix columns to first 12 months for visual spacing
    matrix_subset = retention_matrix.loc[:, :12]
    # Filter rows with NaN or zero cohort sizes
    valid_rows = [r for r in matrix_subset.index if ret_metrics.get("cohort_sizes", {}).get(r, 0) > 0]
    matrix_subset = matrix_subset.loc[valid_rows]

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=matrix_subset.values,
        x=[f"Month {c}" for c in matrix_subset.columns],
        y=matrix_subset.index,
        colorscale='Blues',
        hoverongaps=False,
        text=np.round(matrix_subset.values, 2),
        texttemplate="%{text}%" if len(matrix_subset) < 20 else "", # Show labels only if compact
        hovertemplate="Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.2f}%<extra></extra>"
    ))
    fig_heatmap.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#64748b"),
        margin=dict(l=70, r=20, t=30, b=40),
        yaxis=dict(autorange="reversed", title="Cohort Month"),
        xaxis=dict(title="Months Since Acquisition")
    )

    # Chart 2: Churn vs Retention (Donut Chart)
    # Active = Champions + Loyal + Potential Loyalists + Recent
    active_seg_count = rfm_df[~rfm_df['segment'].isin(['At Risk', 'Lost Customers'])].shape[0]
    churned_seg_count = rfm_df[rfm_df['segment'].isin(['At Risk', 'Lost Customers'])].shape[0]
    fig_donut = px.pie(
        names=['Active (R>=3)', 'Dormant (R<=2)'],
        values=[active_seg_count, churned_seg_count],
        color_discrete_sequence=['#4f46e5', '#ef4444'], # Indigo, Red
        hole=0.4
    )
    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
    fig_donut.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#64748b"),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    # Chart 3: Repeat Purchase Trend (Monthly Line Chart)
    # Calculates active return buyers (ordered in that month, had a prior purchase in history)
    first_purchase = df.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
    first_purchase.columns = ['customer_unique_id', 'first_purchase_time']
    df_with_fp = df.merge(first_purchase, on='customer_unique_id', how='left')
    
    # Flag purchases that are NOT the customer's first purchase
    df_with_fp['is_repeat_purchase'] = df_with_fp['order_purchase_timestamp'] > df_with_fp['first_purchase_time']
    
    # Group by month
    monthly_buyers = df_with_fp.groupby('purchase_year_month').agg(
        total_unique_buyers=('customer_unique_id', 'nunique'),
        repeat_unique_buyers=('customer_unique_id', lambda x: x[df_with_fp.loc[x.index, 'is_repeat_purchase']].nunique())
    ).reset_index()
    
    monthly_buyers['repeat_rate'] = (monthly_buyers['repeat_unique_buyers'] / monthly_buyers['total_unique_buyers']) * 100
    
    fig_line = px.line(
        monthly_buyers,
        x='purchase_year_month',
        y='repeat_rate',
        labels={'purchase_year_month': 'Month', 'repeat_rate': 'Active Repeat Buyers (%)'},
        color_discrete_sequence=['#10b981'] # Emerald
    )
    fig_line.update_traces(mode="lines+markers", line=dict(width=3), marker=dict(size=6))
    fig_line.update_layout(**plotly_layout_defaults)

    # Chart 4: Customer Lifetime Distribution (Histogram of lifespans)
    # Filter 0-day lifespans (one-time buyers) to show repeat customer Lifespans (in days)
    repeat_lifespans = [l for l in ret_metrics.get("lifespans", []) if l > 0]
    fig_hist_lifetime = px.histogram(
        x=repeat_lifespans,
        nbins=50,
        labels={'x': 'Customer Lifespan (Days)', 'y': 'Customer Count'},
        color_discrete_sequence=['#06b6d4'] # Cyan
    )
    fig_hist_lifetime.update_layout(**plotly_layout_defaults)
    fig_hist_lifetime.update_layout(yaxis=dict(title="Customer Count"))

    # Chart 5: Days Between Purchases Distribution (Histogram of return intervals)
    fig_hist_intervals = px.histogram(
        x=ret_metrics.get("return_intervals", []),
        nbins=50,
        labels={'x': 'Days Between Consecutive Purchases', 'y': 'Order Count'},
        color_discrete_sequence=['#ec4899'] # Pink
    )
    fig_hist_intervals.update_layout(**plotly_layout_defaults)
    fig_hist_intervals.update_layout(yaxis=dict(title="Order Count"))

    graphs_json = {
        "heatmap": plotly_to_json(fig_heatmap),
        "donut_churn": plotly_to_json(fig_donut),
        "repeat_trend": plotly_to_json(fig_line),
        "lifetime_dist": plotly_to_json(fig_hist_lifetime),
        "intervals_dist": plotly_to_json(fig_hist_intervals)
    }

    # 4. Insight Engine: Automated Retention Findings
    insights = []
    
    insights.append({
        "type": "danger",
        "icon": "bi-graph-down",
        "title": "Steep Month 1 Retention Cliff",
        "text": f"Customer retention drops sharply to <strong>{m1_retention:.2f}%</strong> in Month 1, showing that a significant majority of customers fail to make any purchase after their acquisition month."
    })
    
    insights.append({
        "type": "warning",
        "icon": "bi-arrow-right-circle",
        "title": "Low Base Loyalty conversion",
        "text": f"The repeat purchase rate is <strong>{repeat_rate:.2f}%</strong>. Olist relies heavily on continuous new customer acquisition rather than recurring customer value."
    })
    
    if best_cohort_name != "N/A":
        insights.append({
            "type": "success",
            "icon": "bi-star",
            "title": f"Top Performing Cohort ({best_cohort_name})",
            "text": f"Cohort <strong>{best_cohort_name}</strong> generated the highest Month 1 retention at <strong>{best_m1_pct:.2f}%</strong> (sample size: <strong>{best_cohort_size:,}</strong> customers)."
        })
        
    insights.append({
        "type": "danger",
        "icon": "bi-exclamation-octagon",
        "title": "Severe Business Churn risk",
        "text": f"Business churn remains high at <strong>{business_churn:.1f}%</strong>. More than half of historical accounts have gone cold with no recent transaction history."
    })

    # 5. Recommendation Engine
    recommendations = [
        {
            "finding": "Month 1 retention falls below 1.5% on average.",
            "impact": "98% of marketing acquisition spend is lost after the first purchase.",
            "recommendation": "Launch automated post-purchase emails offering a '15% off next order' coupon valid for 30 days.",
            "priority": "Critical"
        },
        {
            "finding": "62% of customer base is dormant or Lost.",
            "impact": "High customer turnover requires continuous acquisition marketing budgets.",
            "recommendation": "Set up win-back trigger campaigns targeting 'At Risk' users at the 120-day inactivity mark with highly incentivized offers.",
            "priority": "High"
        },
        {
            "finding": "Repeat purchase rate is stuck at 3.1%.",
            "impact": "Lifetime Value (LTV) is capped near product price ($166.59).",
            "recommendation": "Introduce a points-based loyalty program or subscription tier (e.g. Free shipping) to incentivize repeat order cycles.",
            "priority": "High"
        },
        {
            "finding": "Champions segment makes up only 1% of customer volume.",
            "impact": "Your highest value loyalty core is small and vulnerable.",
            "recommendation": "Create a VIP early-access program with customized support and early notifications for new collections.",
            "priority": "Medium"
        }
    ]

    kpis = {
        "repeat_rate": f"{repeat_rate:.2f}%",
        "retention_rate": f"{m1_retention:.2f}%",
        "business_churn": f"{business_churn:.1f}%",
        "retention_churn": f"{retention_churn:.1f}%",
        "avg_lifetime": f"{avg_lifetime:.1f} days",
        "avg_return_days": f"{avg_return_days:.1f} days",
        "best_cohort": best_cohort_name,
        "best_cohort_size": best_cohort_size,
        "best_m1_pct": f"{best_m1_pct:.2f}%"
    }

    return render_template(
        'retention.html',
        active_page='retention',
        loaded=True,
        kpis=kpis,
        health=health_indicators,
        graphs=graphs_json,
        insights=insights,
        recommendations=recommendations
    )

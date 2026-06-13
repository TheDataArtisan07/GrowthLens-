from flask import Blueprint, render_template
from services.data_manager import data_manager
import analytics.customers as cust_calc
import plotly.express as px
import plotly.graph_objects as go
from growthlens.utils import plotly_to_json
import json
import pandas as pd
import numpy as np

customers_bp = Blueprint('customers', __name__)

@customers_bp.route('/customers')
def index():
    df = data_manager.get_analytics_df()
    rfm_df = data_manager.get_rfm_df()
    diagnostics = data_manager.get_diagnostics()

    if df is None or rfm_df is None or not diagnostics.get("load_success", False):
        return render_template('customers.html', active_page='customers', loaded=False)

    # 1. Core KPIs
    total_customers = cust_calc.calculate_total_customers(df)
    repeat_customers = cust_calc.calculate_repeat_customers(df)
    one_time_customers = cust_calc.calculate_one_time_customers(df)
    
    repeat_rate = 0.0
    if total_customers > 0:
        repeat_rate = (repeat_customers / total_customers) * 100
        
    avg_rev_per_customer = cust_calc.customer_lifetime_metrics(df)

    # Find dominant customer segment
    top_segment_name = "N/A"
    top_segment_count = 0
    if not rfm_df.empty:
        seg_counts = rfm_df['segment'].value_counts()
        if not seg_counts.empty:
            top_segment_name = seg_counts.index[0]
            top_segment_count = int(seg_counts.iloc[0])

    # 2. Build Customer Health Indicators
    # A. Retention Indicator
    if repeat_rate >= 15.0:
        retention_status = "Healthy"
        retention_badge = "success"
        retention_desc = "High repurchase velocity; customer loyalty matches top-tier SaaS standards."
    elif repeat_rate >= 5.0:
        retention_status = "Moderate"
        retention_badge = "warning"
        retention_desc = "Steady secondary purchase rate, but room to optimize reactivation campaigns."
    else:
        retention_status = "Needs Attention"
        retention_badge = "danger"
        retention_desc = "Extremely low repeat purchase rate (~3%). The customer base is heavily transactional."

    # B. Concentration Indicator (Pareto check: top 10% customers vs revenue)
    sorted_monetary = rfm_df['monetary'].sort_values(ascending=False)
    top_10_count = max(1, int(len(sorted_monetary) * 0.1))
    top_10_rev = sorted_monetary.head(top_10_count).sum()
    total_payment_rev = sorted_monetary.sum()
    
    concentration_pct = 0.0
    if total_payment_rev > 0:
        concentration_pct = (top_10_rev / total_payment_rev) * 100

    if concentration_pct >= 80.0:
        concentration_status = "Needs Attention"
        concentration_badge = "danger"
        concentration_desc = f"Top 10% of customers generate {concentration_pct:.1f}% of revenue. High client risk."
    elif concentration_pct >= 50.0:
        concentration_status = "Moderate"
        concentration_badge = "warning"
        concentration_desc = f"Top 10% of customers generate {concentration_pct:.1f}% of revenue. Moderate risk exposure."
    else:
        concentration_status = "Healthy"
        concentration_badge = "success"
        concentration_desc = f"Top 10% of customers generate {concentration_pct:.1f}% of revenue. Revenue is widely distributed."

    # C. Segment Diversity Indicator (Largest segment concentration)
    max_segment_pct = 0.0
    if len(rfm_df) > 0:
        max_segment_pct = (top_segment_count / len(rfm_df)) * 100

    if max_segment_pct >= 70.0:
        diversity_status = "Needs Attention"
        diversity_badge = "danger"
        diversity_desc = f"Largest segment ({top_segment_name}) holds {max_segment_pct:.1f}% of customers. Highly skewed."
    elif max_segment_pct >= 40.0:
        diversity_status = "Moderate"
        diversity_badge = "warning"
        diversity_desc = f"Largest segment holds {max_segment_pct:.1f}% of customer counts. Standard distribution."
    else:
        diversity_status = "Healthy"
        diversity_badge = "success"
        diversity_desc = f"Largest segment holds {max_segment_pct:.1f}% of customer counts. High segment diversity."

    health_indicators = {
        "retention": {"status": retention_status, "badge": retention_badge, "desc": retention_desc},
        "concentration": {"status": concentration_status, "badge": concentration_badge, "desc": concentration_desc},
        "diversity": {"status": diversity_status, "badge": diversity_badge, "desc": diversity_desc}
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

    # Chart 1: Customer Segment Distribution (Pie Chart)
    seg_counts_df = rfm_df['segment'].value_counts().reset_index()
    seg_counts_df.columns = ['segment', 'customer_count']
    fig_segment_pie = px.pie(
        seg_counts_df,
        values='customer_count',
        names='segment',
        color_discrete_sequence=px.colors.qualitative.Safe,
        hole=0.4
    )
    fig_segment_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_segment_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#64748b"),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    # Chart 2: Customer Distribution by State (Bar Chart, Top 20)
    state_dist = cust_calc.customer_distribution_by_state(df).head(20)
    fig_state = px.bar(
        state_dist,
        x='customer_state',
        y='customer_count',
        labels={'customer_state': 'State', 'customer_count': 'Unique Customers'},
        color_discrete_sequence=['#4f46e5'] # Indigo
    )
    fig_state.update_layout(**plotly_layout_defaults)

    # Chart 3: Purchase Frequency Distribution (Histogram/Bar Chart log scale)
    freq_dist = cust_calc.customer_purchase_frequency(df)
    # Filter/group higher order counts for clean visualization
    freq_dist_limited = freq_dist.copy()
    freq_dist_limited['order_count_label'] = freq_dist_limited['order_count'].astype(str) + " Order"
    freq_dist_limited.loc[freq_dist_limited['order_count'] >= 4, 'order_count_label'] = "4+ Orders"
    freq_grouped = freq_dist_limited.groupby('order_count_label')['customer_count'].sum().reset_index()
    
    # Sort order count label logically
    sort_order = {"1 Order": 0, "2 Orders": 1, "3 Orders": 2, "4+ Orders": 3}
    freq_grouped['sort_idx'] = freq_grouped['order_count_label'].map(sort_order).fillna(4)
    freq_grouped = freq_grouped.sort_values('sort_idx')

    fig_freq = px.bar(
        freq_grouped,
        x='order_count_label',
        y='customer_count',
        labels={'order_count_label': 'Number of Orders placed', 'customer_count': 'Customer Count'},
        color_discrete_sequence=['#10b981'] # Emerald
    )
    fig_freq.update_layout(**plotly_layout_defaults)
    fig_freq.update_layout(yaxis=dict(type='log', title="Customer Count (Log Scale)")) # Log scale to show repeats clearly

    # Chart 4: Top 20 Customers by Revenue (Horizontal Bar Chart)
    top_custs = cust_calc.top_customers_by_revenue(df)
    top_custs['customer_label'] = "Cust " + top_custs['customer_unique_id'].str[:8] + "..."
    fig_top_cust = px.bar(
        top_custs,
        x='payment_value',
        y='customer_label',
        orientation='h',
        labels={'payment_value': 'Total Spent ($)', 'customer_label': 'Customer Unique ID'},
        color_discrete_sequence=['#06b6d4'],
        hover_data={'customer_unique_id': True, 'payment_value': ':.2f'}
    )
    fig_top_cust.update_layout(**plotly_layout_defaults)
    fig_top_cust.update_layout(yaxis=dict(autorange="reversed"))

    # Chart 5: Revenue Contribution by Customer Segment (Bar Chart)
    seg_rev = rfm_df.groupby('segment')['monetary'].sum().reset_index()
    seg_rev = seg_rev.sort_values(by='monetary', ascending=False)
    fig_seg_rev = px.bar(
        seg_rev,
        x='segment',
        y='monetary',
        labels={'segment': 'RFM Customer Segment', 'monetary': 'Revenue Contribution ($)'},
        color_discrete_sequence=['#ec4899'] # Pink
    )
    fig_seg_rev.update_layout(**plotly_layout_defaults)

    graphs_json = {
        "segment_pie": plotly_to_json(fig_segment_pie),
        "customer_state": plotly_to_json(fig_state),
        "purchase_frequency": plotly_to_json(fig_freq),
        "top_customers": plotly_to_json(fig_top_cust),
        "segment_revenue": plotly_to_json(fig_seg_rev)
    }

    # 4. Insight Engine: Customer Findings
    insights = []
    
    # Pareto insight (Concentration check)
    insights.append({
        "type": "info",
        "icon": "bi-pie-chart-fill",
        "title": "Pareto Distribution Check",
        "text": f"The top 10% of customers by spending generate <strong>{concentration_pct:.1f}%</strong> of total revenue, representing a low concentration risk compared to standard enterprise B2B environments."
    })

    # One-Time Purchase dominance
    one_time_pct = (one_time_customers / total_customers) * 100 if total_customers > 0 else 0.0
    insights.append({
        "type": "warning",
        "icon": "bi-exclamation-triangle",
        "title": "Single-Purchase Dominance",
        "text": f"A staggering <strong>{one_time_pct:.1f}%</strong> of customers have placed only 1 order, showing that growth is driven primarily by new user acquisition rather than repeats."
    })

    # Champions Segment Contribution
    champions_df = rfm_df[rfm_df['segment'] == 'Champions']
    champions_count = len(champions_df)
    champions_pct = (champions_count / len(rfm_df)) * 100 if len(rfm_df) > 0 else 0.0
    champions_rev = champions_df['monetary'].sum()
    champions_rev_pct = (champions_rev / total_payment_rev) * 100 if total_payment_rev > 0 else 0.0
    
    insights.append({
        "type": "success",
        "icon": "bi-award",
        "title": "Champions Contribution",
        "text": f"Champions make up only <strong>{champions_pct:.2f}%</strong> of the buyer base but generate <strong>{champions_rev_pct:.1f}%</strong> of revenue, representing your highest LTV segment."
    })

    # Needs Attention / At Risk segments
    at_risk_df = rfm_df[rfm_df['segment'].isin(['At Risk', 'Lost Customers'])]
    at_risk_pct = (len(at_risk_df) / len(rfm_df)) * 100 if len(rfm_df) > 0 else 0.0
    insights.append({
        "type": "danger",
        "icon": "bi-person-dash",
        "title": "Churn & At-Risk Concentration",
        "text": f"<strong>{at_risk_pct:.1f}%</strong> of historical customers are classified as Lost or At Risk, indicating a need for targetted push notifications and discount reactivations."
    })

    kpis = {
        "total_customers": total_customers,
        "repeat_customers": repeat_customers,
        "one_time_customers": one_time_customers,
        "repeat_rate": f"{repeat_rate:.2f}%",
        "avg_rev_per_customer": avg_rev_per_customer,
        "top_segment": top_segment_name
    }

    return render_template(
        'customers.html',
        active_page='customers',
        loaded=True,
        kpis=kpis,
        health=health_indicators,
        graphs=graphs_json,
        insights=insights
    )

from flask import Blueprint, render_template
from services.data_manager import data_manager
import analytics.revenue as rev_calc
import analytics.customers as cust_calc
import analytics.retention as ret_calc
import analytics.reviews as rev_sat_calc
import analytics.action_center as ac_calc
import plotly.express as px
import plotly.graph_objects as go
from growthlens.utils import plotly_to_json
import json
import pandas as pd
import numpy as np

action_center_bp = Blueprint('action_center', __name__)

@action_center_bp.route('/action-center')
def index():
    df = data_manager.get_analytics_df()
    rfm_df = data_manager.get_rfm_df()
    retention_matrix = data_manager.get_retention_matrix()
    diagnostics = data_manager.get_diagnostics()

    if df is None or rfm_df is None or retention_matrix is None or not diagnostics.get("load_success", False):
        return render_template('action_center.html', active_page='action_center', loaded=False)

    # 1. Fetch calculations
    snapshot = ac_calc.get_action_center_snapshot(df, retention_matrix)
    trends = ac_calc.get_quick_metrics_trends(df)
    
    reviews_metrics = rev_sat_calc.customer_satisfaction_metrics(df)
    findings = ac_calc.identify_top_findings(df, rfm_df, retention_matrix, reviews_metrics)
    working_well = ac_calc.get_working_well(df, rfm_df, retention_matrix, reviews_metrics)
    needs_attention = ac_calc.get_needs_attention(df, rfm_df, retention_matrix, reviews_metrics)
    
    actions = ac_calc.get_recommended_actions(df, rfm_df, retention_matrix, reviews_metrics)
    opportunities = ac_calc.get_top_opportunities(df, rfm_df, retention_matrix)
    summary = ac_calc.generate_executive_summary(df, rfm_df, retention_matrix, reviews_metrics)

    # 2. Build Charts (Max 2)
    # Chart 1: Revenue Trend Line Chart
    monthly_rev = rev_calc.monthly_revenue_trend(df)
    # Clean up months with low transaction volume (<100 orders) to match other reports
    monthly_rev = monthly_rev[monthly_rev['purchase_year_month'].isin(
        df.groupby('purchase_year_month')['order_id'].nunique().loc[lambda x: x >= 100].index
    )]
    
    fig_rev = px.line(
        monthly_rev,
        x='purchase_year_month',
        y='payment_value',
        labels={'purchase_year_month': 'Month', 'payment_value': 'Revenue ($)'},
        color_discrete_sequence=['#4f46e5'] # Premium Indigo
    )
    fig_rev.update_traces(line=dict(width=3), mode='lines+markers', marker=dict(size=6))
    fig_rev.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#64748b"),
        margin=dict(l=40, r=20, t=10, b=20),
        xaxis=dict(showgrid=False, linecolor="#cbd5e1"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1")
    )

    # Chart 2: Customer Segment Distribution Donut Chart
    seg_counts = rfm_df.groupby('segment')['customer_unique_id'].count().reset_index()
    seg_counts.columns = ['segment', 'customer_count']
    
    fig_seg = px.pie(
        seg_counts,
        values='customer_count',
        names='segment',
        hole=0.5,
        color_discrete_sequence=['#4f46e5', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#64748b']
    )
    fig_seg.update_traces(textinfo='percent', hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}")
    fig_seg.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#64748b"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )

    graphs_json = {
        "revenue_trend": plotly_to_json(fig_rev),
        "segment_dist": plotly_to_json(fig_seg)
    }

    return render_template(
        'action_center.html',
        active_page='action_center',
        loaded=True,
        snapshot=snapshot,
        trends=trends,
        findings=findings,
        working_well=working_well,
        needs_attention=needs_attention,
        actions=actions,
        opportunities=opportunities,
        summary=summary,
        graphs=graphs_json
    )

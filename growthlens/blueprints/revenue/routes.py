from flask import Blueprint, render_template
from services.data_manager import data_manager
import analytics.revenue as rev_calc
import plotly.express as px
import plotly.graph_objects as go
from growthlens.utils import plotly_to_json
import json
import pandas as pd
import numpy as np

revenue_bp = Blueprint('revenue', __name__)

@revenue_bp.route('/revenue')
def index():
    df = data_manager.get_analytics_df()
    diagnostics = data_manager.get_diagnostics()
    
    if df is None or not diagnostics.get("load_success", False):
        # Fallback if DataManager failed or CSVs not found
        return render_template('revenue.html', active_page='revenue', loaded=False)

    # 1. Calculate Core KPI Metrics
    total_revenue = rev_calc.calculate_total_revenue(df)
    avg_order_value = rev_calc.calculate_average_order_value(df)
    total_orders = rev_calc.calculate_total_orders(df)
    
    # Category splits
    cat_rev = rev_calc.revenue_by_category(df)
    top_category_name = "N/A"
    top_category_rev = 0.0
    if not cat_rev.empty:
        top_category_name = cat_rev.iloc[0]['product_category_name_english']
        top_category_rev = cat_rev.iloc[0]['order_value']
        
    # State splits
    state_rev = rev_calc.revenue_by_state(df)
    top_state_name = "N/A"
    top_state_rev = 0.0
    if not state_rev.empty:
        top_state_name = state_rev.iloc[0]['customer_state']
        top_state_rev = state_rev.iloc[0]['payment_value']

    # Monthly Growth Rate Table
    monthly_growth = rev_calc.monthly_growth_rate(df)
    growth_rate_display = "N/A"
    mom_value = 0.0
    if len(monthly_growth) > 1:
        # Get latest full month growth
        latest_row = monthly_growth.iloc[-1]
        mom_value = latest_row['growth_rate']
        growth_rate_display = f"{mom_value:+.2f}%"

    # 2. Build Plotly Visualizations
    plotly_layout_defaults = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#64748b"),
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        hovermode="closest"
    )

    # Chart 1: Monthly Revenue Trend (Line Chart)
    monthly_trend = rev_calc.monthly_revenue_trend(df)
    fig_monthly = px.line(
        monthly_trend, 
        x='purchase_year_month', 
        y='payment_value',
        labels={'purchase_year_month': 'Month', 'payment_value': 'Revenue ($)'},
        color_discrete_sequence=['#4f46e5'] # Indigo 600
    )
    fig_monthly.update_traces(mode="lines+markers", line=dict(width=3), marker=dict(size=6))
    fig_monthly.update_layout(**plotly_layout_defaults)
    
    # Chart 2: Revenue by Product Category (Top 15 Bar Chart)
    top15_categories = cat_rev.head(15)
    fig_category = px.bar(
        top15_categories,
        x='product_category_name_english',
        y='order_value',
        labels={'product_category_name_english': 'Category', 'order_value': 'Sales ($)'},
        color_discrete_sequence=['#6366f1'] # Violet Accent
    )
    fig_category.update_layout(**plotly_layout_defaults)
    fig_category.update_layout(xaxis=dict(tickangle=45))

    # Chart 3: Revenue by State (Bar Chart)
    fig_state = px.bar(
        state_rev.head(20), # Limit to top 20 states for cleanliness
        x='customer_state',
        y='payment_value',
        labels={'customer_state': 'State', 'payment_value': 'Revenue ($)'},
        color_discrete_sequence=['#06b6d4'] # Cyan Accent
    )
    fig_state.update_layout(**plotly_layout_defaults)

    # Chart 4: Top 20 Products/Categories by Revenue (Horizontal Bar Chart)
    top20_cats = cat_rev.head(20)
    fig_top_prod = px.bar(
        top20_cats,
        x='order_value',
        y='product_category_name_english',
        orientation='h',
        labels={'order_value': 'Revenue ($)', 'product_category_name_english': 'Category'},
        color_discrete_sequence=['#10b981'] # Emerald Accent
    )
    fig_top_prod.update_layout(**plotly_layout_defaults)
    fig_top_prod.update_layout(yaxis=dict(autorange="reversed")) # High-low descending top to bottom

    # Chart 5: Average Order Value Trend (Line Chart)
    monthly_aov = rev_calc.monthly_aov_trend(df)
    fig_aov = px.line(
        monthly_aov,
        x='purchase_year_month',
        y='aov',
        labels={'purchase_year_month': 'Month', 'aov': 'Average Order Value ($)'},
        color_discrete_sequence=['#ec4899'] # Pink Accent
    )
    fig_aov.update_traces(mode="lines+markers", line=dict(width=3, dash='solid'), marker=dict(size=6))
    fig_aov.update_layout(**plotly_layout_defaults)

    # Serialize plots to JSON for plotly.js template ingestion
    graphs_json = {
        "monthly_revenue": plotly_to_json(fig_monthly),
        "revenue_category": plotly_to_json(fig_category),
        "revenue_state": plotly_to_json(fig_state),
        "top_products": plotly_to_json(fig_top_prod),
        "average_order": plotly_to_json(fig_aov)
    }

    # 3. Insight Engine: Automated Revenue Findings
    insights = []
    
    # MoM Revenue growth insight
    if len(monthly_growth) > 1:
        latest_month = monthly_growth.iloc[-1]['purchase_year_month']
        if mom_value > 0:
            insights.append({
                "type": "success",
                "icon": "bi-graph-up-arrow",
                "title": f"Growth Momentum in {latest_month}",
                "text": f"Monthly revenue increased by <strong>{mom_value:.1f}%</strong> compared to the preceding month, signalling strong transaction momentum."
            })
        elif mom_value < 0:
            insights.append({
                "type": "warning",
                "icon": "bi-graph-down-arrow",
                "title": f"Revenue Dip in {latest_month}",
                "text": f"Monthly sales revenue dropped by <strong>{abs(mom_value):.1f}%</strong> MoM, indicating a potential seasonal contraction."
            })

    # Category contribution insight
    total_item_revenue = cat_rev['order_value'].sum()
    if total_item_revenue > 0 and not cat_rev.empty:
        contrib_pct = (top_category_rev / total_item_revenue) * 100
        insights.append({
            "type": "info",
            "icon": "bi-box-seam",
            "title": f"Dominant Product Category",
            "text": f"The top category, <strong>{top_category_name.replace('_', ' ')}</strong>, generates <strong>${top_category_rev:,.2f}</strong>, contributing <strong>{contrib_pct:.1f}%</strong> of total product value."
        })

    # State concentration insight
    total_payment_revenue = state_rev['payment_value'].sum()
    if total_payment_revenue > 0 and not state_rev.empty:
        state_pct = (top_state_rev / total_payment_revenue) * 100
        insights.append({
            "type": "info",
            "icon": "bi-geo-alt",
            "title": "Geographic Consolidation",
            "text": f"State <strong>{top_state_name}</strong> is the primary revenue engine, generating <strong>${top_state_rev:,.2f}</strong> (<strong>{state_pct:.1f}%</strong> of all order value)."
        })

    # AOV trend direction
    if len(monthly_aov) >= 3:
        # Check trend direction of the last 3 months
        last_3 = monthly_aov.tail(3)['aov'].tolist()
        if last_3[2] > last_3[0]:
            insights.append({
                "type": "success",
                "icon": "bi-arrow-up-right-circle",
                "title": "AOV Growth Trend",
                "text": f"Average Order Value has increased over the last 3 months to <strong>${last_3[2]:.2f}</strong>, showing higher buyer basket sizes."
            })
        else:
            insights.append({
                "type": "neutral",
                "icon": "bi-arrow-right-circle",
                "title": "AOV Stability",
                "text": f"Average order basket values are hovering stably near <strong>${avg_order_value:.2f}</strong> across recent billing cycles."
            })

    kpis = {
        "total_revenue": total_revenue,
        "average_order_value": avg_order_value,
        "total_orders": total_orders,
        "growth_rate": growth_rate_display,
        "top_category": top_category_name.replace('_', ' '),
        "top_state": top_state_name
    }

    return render_template(
        'revenue.html', 
        active_page='revenue', 
        loaded=True,
        kpis=kpis, 
        graphs=graphs_json,
        insights=insights
    )

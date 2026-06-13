from flask import Blueprint, render_template
from services.data_manager import data_manager
import analytics.forecasting as fore_calc
import plotly.express as px
import plotly.graph_objects as go
from growthlens.utils import plotly_to_json
import json
import pandas as pd
import numpy as np

forecasting_bp = Blueprint('forecasting', __name__)

@forecasting_bp.route('/forecasting')
def index():
    df = data_manager.get_analytics_df()
    rfm_df = data_manager.get_rfm_df()
    retention_matrix = data_manager.get_retention_matrix()
    diagnostics = data_manager.get_diagnostics()

    if df is None or rfm_df is None or retention_matrix is None or not diagnostics.get("load_success", False):
        return render_template('forecasting.html', active_page='forecasting', loaded=False)

    # 1. Compute predictions
    revenue_forecast = fore_calc.get_revenue_forecast(df)
    customer_forecast = fore_calc.get_customer_growth_forecast(df)
    churn_risk = fore_calc.get_churn_risk_analysis(rfm_df)
    category_outlook = fore_calc.get_category_outlook(df)
    
    summary = fore_calc.generate_forecast_summary(df, rfm_df, revenue_forecast, customer_forecast)
    indicators = fore_calc.get_predictive_health_indicators(revenue_forecast, customer_forecast)

    # 2. Build Charts
    plotly_layout_defaults = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#64748b"),
        margin=dict(l=50, r=20, t=10, b=30),
        xaxis=dict(showgrid=False, linecolor="#cbd5e1"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        hovermode="closest"
    )

    # Chart 1: Revenue Forecast Line Chart
    hist_rev = revenue_forecast["history"]
    fore_rev = revenue_forecast["forecast"]
    
    hist_x = [pt["purchase_year_month"] for pt in hist_rev]
    hist_y = [pt["payment_value"] for pt in hist_rev]
    
    fore_x = [pt["purchase_year_month"] for pt in fore_rev]
    fore_y = [pt["payment_value"] for pt in fore_rev]
    
    # Prepend last point to connect lines
    if hist_x and fore_x:
        connect_x = [hist_x[-1]] + fore_x
        connect_y = [hist_y[-1]] + fore_y
    else:
        connect_x, connect_y = fore_x, fore_y

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        name="Historical Revenue",
        mode="lines+markers",
        line=dict(color='#4f46e5', width=3),
        marker=dict(size=6)
    ))
    fig_rev.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        name="Projected Forecast",
        mode="lines+markers",
        line=dict(color='#8b5cf6', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    fig_rev.update_layout(**plotly_layout_defaults)
    fig_rev.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Chart 2: Customer Growth Forecast Line Chart
    hist_cust = customer_forecast["history"]
    fore_cust = customer_forecast["forecast"]
    
    hc_x = [pt["purchase_year_month"] for pt in hist_cust]
    hc_new = [pt["new_customers"] for pt in hist_cust]
    hc_rep = [pt["repeat_customers"] for pt in hist_cust]
    
    fc_x = [pt["purchase_year_month"] for pt in fore_cust]
    fc_new = [pt["new_customers"] for pt in fore_cust]
    fc_rep = [pt["repeat_customers"] for pt in fore_cust]
    
    # Prep connections
    if hc_x and fc_x:
        conn_new_x = [hc_x[-1]] + fc_x
        conn_new_y = [hc_new[-1]] + fc_new
        conn_rep_x = [hc_x[-1]] + fc_x
        conn_rep_y = [hc_rep[-1]] + fc_rep
    else:
        conn_new_x, conn_new_y = fc_x, fc_new
        conn_rep_x, conn_rep_y = fc_x, fc_rep

    fig_cust = go.Figure()
    # Acquisitions
    fig_cust.add_trace(go.Scatter(
        x=hc_x, y=hc_new,
        name="Historical Acq",
        mode="lines",
        line=dict(color='#4f46e5', width=2.5)
    ))
    fig_cust.add_trace(go.Scatter(
        x=conn_new_x, y=conn_new_y,
        name="Projected Acq",
        mode="lines",
        line=dict(color='#4f46e5', width=2.5, dash='dash')
    ))
    # Repeats
    fig_cust.add_trace(go.Scatter(
        x=hc_x, y=hc_rep,
        name="Historical Repeat",
        mode="lines",
        line=dict(color='#06b6d4', width=2.5)
    ))
    fig_cust.add_trace(go.Scatter(
        x=conn_rep_x, y=conn_rep_y,
        name="Projected Repeat",
        mode="lines",
        line=dict(color='#06b6d4', width=2.5, dash='dash')
    ))
    fig_cust.update_layout(**plotly_layout_defaults)
    fig_cust.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Chart 3: Churn Risk Donut Chart
    risk_labels = ["Low Risk", "Medium Risk", "High Risk"]
    risk_values = [churn_risk[r]["count"] for r in risk_labels]
    
    fig_risk = px.pie(
        names=risk_labels,
        values=risk_values,
        hole=0.5,
        color=risk_labels,
        color_discrete_map={
            "Low Risk": "#10b981", # Emerald
            "Medium Risk": "#f59e0b", # Amber
            "High Risk": "#ef4444" # Red
        }
    )
    fig_risk.update_traces(textinfo='percent', hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}")
    fig_risk.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#64748b"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )

    # Chart 4: Category Growth Forecast Bar Chart
    growing_cats = category_outlook.get("growing", [])
    weakening_cats = category_outlook.get("weakening", [])
    
    cat_names = [c["category"] for c in growing_cats + weakening_cats]
    cat_slopes = [c["slope"] for c in growing_cats + weakening_cats]
    cat_colors = ['#10b981'] * len(growing_cats) + ['#ef4444'] * len(weakening_cats)
    
    fig_cat = go.Figure(go.Bar(
        x=cat_names,
        y=cat_slopes,
        marker_color=cat_colors,
        hovertemplate="<b>%{x}</b><br>Projected Slope: %{y:,.1f}/mo<extra></extra>"
    ))
    fig_cat.update_layout(**plotly_layout_defaults)
    fig_cat.update_layout(xaxis=dict(tickangle=30))

    graphs_json = {
        "revenue_forecast": plotly_to_json(fig_rev),
        "customer_forecast": plotly_to_json(fig_cust),
        "churn_risk": plotly_to_json(fig_risk),
        "category_outlook": plotly_to_json(fig_cat)
    }

    # Format snapshot indicators
    rev_slope = revenue_forecast["slope"]
    revenue_outlook_snapshot = {
        "slope_val": f"${abs(rev_slope):,.2f}",
        "direction": revenue_forecast["direction"],
        "conf_score": revenue_forecast["confidence_score"],
        "conf_rating": revenue_forecast["confidence_rating"],
        "next_projected": revenue_forecast["forecast"][0]["payment_value"] if revenue_forecast.get("forecast") else 0.0
    }

    return render_template(
        'forecasting.html',
        active_page='forecasting',
        loaded=True,
        revenue_forecast=revenue_forecast,
        revenue_snapshot=revenue_outlook_snapshot,
        customer_forecast=customer_forecast,
        churn_risk=churn_risk,
        category_outlook=category_outlook,
        summary=summary,
        indicators=indicators,
        graphs=graphs_json
    )

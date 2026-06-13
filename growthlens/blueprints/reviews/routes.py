from flask import Blueprint, render_template
from services.data_manager import data_manager
import analytics.reviews as rev_calc
import plotly.express as px
import plotly.graph_objects as go
from growthlens.utils import plotly_to_json
import json
import pandas as pd
import numpy as np

reviews_bp = Blueprint('reviews', __name__)

@reviews_bp.route('/reviews')
def index():
    df = data_manager.get_analytics_df()
    diagnostics = data_manager.get_diagnostics()

    if df is None or not diagnostics.get("load_success", False):
        return render_template('reviews.html', active_page='reviews', loaded=False)

    # 1. Calculate Customer Satisfaction KPIs
    metrics = rev_calc.customer_satisfaction_metrics(df)
    ret_sat = rev_calc.satisfaction_by_retention(df)

    # 2. Satisfaction & Delivery Health Indicators
    # A. Customer Satisfaction
    avg_rating = metrics["avg_review_score"]
    if avg_rating >= 4.0:
        sat_status = "Healthy"
        sat_badge = "success"
        sat_desc = f"Average review score is {avg_rating:.2f} / 5.00. Customers are highly satisfied."
    elif avg_rating >= 3.5:
        sat_status = "Moderate"
        sat_badge = "warning"
        sat_desc = f"Average review score is {avg_rating:.2f} / 5.00. Stable, but monitoring needed."
    else:
        sat_status = "Needs Attention"
        sat_badge = "danger"
        sat_desc = f"Average review score is {avg_rating:.2f} / 5.00. High customer friction."

    # B. Delivery Performance
    ontime_rate = metrics["on_time_delivery_rate"]
    if ontime_rate >= 90.0:
        del_status = "Healthy"
        del_badge = "success"
        del_desc = f"On-time delivery rate is {ontime_rate:.1f}%. High operational delivery standard."
    elif ontime_rate >= 80.0:
        del_status = "Moderate"
        del_badge = "warning"
        del_desc = f"On-time delivery rate is {ontime_rate:.1f}%. Stable logistics but minor delays."
    else:
        del_status = "Needs Attention"
        del_badge = "danger"
        del_desc = f"On-time delivery rate is {ontime_rate:.1f}%. Severe shipping bottlenecks."

    # C. Review Sentiment
    pos_rate = metrics["positive_rate"]
    if pos_rate >= 70.0:
        sent_status = "Healthy"
        sent_badge = "success"
        sent_desc = f"Positive reviews make up {pos_rate:.1f}% of feedback. High customer advocacy."
    elif pos_rate >= 50.0:
        sent_status = "Moderate"
        sent_badge = "warning"
        sent_desc = f"Positive reviews make up {pos_rate:.1f}% of feedback. Balanced sentiment split."
    else:
        sent_status = "Needs Attention"
        sent_badge = "danger"
        sent_desc = f"Positive reviews are only {pos_rate:.1f}%. Customer feedback is heavily skewed negative."

    health_indicators = {
        "satisfaction": {"status": sat_status, "badge": sat_badge, "desc": sat_desc},
        "delivery": {"status": del_status, "badge": del_badge, "desc": del_desc},
        "sentiment": {"status": sent_status, "badge": sent_badge, "desc": sent_desc}
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

    # Chart 1: Review Score Distribution (Histogram)
    dist_dict = rev_calc.review_score_distribution(df)
    fig_dist = px.bar(
        x=list(dist_dict.keys()),
        y=list(dist_dict.values()),
        labels={'x': 'Review Score', 'y': 'Review Count'},
        color_discrete_sequence=['#4f46e5'] # Indigo
    )
    fig_dist.update_layout(**plotly_layout_defaults)
    fig_dist.update_layout(xaxis=dict(dtick=1))

    # Chart 2: Review Sentiment Distribution (Donut Chart)
    fig_donut = px.pie(
        names=['Positive (4-5)', 'Neutral (3)', 'Negative (1-2)'],
        values=[
            dist_dict.get(4, 0) + dist_dict.get(5, 0),
            dist_dict.get(3, 0),
            dist_dict.get(1, 0) + dist_dict.get(2, 0)
        ],
        color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444'], # Emerald, Amber, Red
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

    # Chart 3: Average Rating by Product Category (Top 15 Bar Chart)
    cat_reviews = rev_calc.reviews_by_category(df).head(15)
    fig_cat = px.bar(
        cat_reviews,
        x='product_category_name_english',
        y='avg_review_score',
        labels={'product_category_name_english': 'Category', 'avg_review_score': 'Avg Rating'},
        color_discrete_sequence=['#8b5cf6'] # Purple
    )
    fig_cat.update_layout(**plotly_layout_defaults)
    fig_cat.update_layout(xaxis=dict(tickangle=45), yaxis=dict(range=[1, 5]))

    # Chart 4: Delivery Days vs Review Score (Box Plot)
    corr_df = rev_calc.delivery_vs_review_correlation(df)
    if len(corr_df) > 5000:
        corr_df = corr_df.sample(5000, random_state=42)
    corr_df['review_score_label'] = corr_df['review_score'].astype(str) + " Star"
    fig_box = px.box(
        corr_df,
        x='review_score_label',
        y='delivery_days',
        labels={'review_score_label': 'Review Rating', 'delivery_days': 'Delivery Time (Days)'},
        color_discrete_sequence=['#6366f1'] # Violet
    )
    fig_box.update_layout(**plotly_layout_defaults)

    # Chart 5: Late Delivery Impact Analysis (Bar Chart)
    # Filter unique orders that have reviews and actual delivery/estimated dates
    order_df = df.drop_duplicates(subset=['order_id']).dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date', 'review_score'])
    late_flags = order_df['order_delivered_customer_date'] > order_df['order_estimated_delivery_date']
    ontime_avg = float(order_df[~late_flags]['review_score'].mean()) if not order_df[~late_flags].empty else 0.0
    late_avg = float(order_df[late_flags]['review_score'].mean()) if not order_df[late_flags].empty else 0.0

    fig_impact = px.bar(
        x=['On-Time Deliveries', 'Late Deliveries'],
        y=[ontime_avg, late_avg],
        labels={'x': 'Delivery Status', 'y': 'Average Review Rating'},
        color=['On-Time', 'Late'],
        color_discrete_map={'On-Time': '#10b981', 'Late': '#ef4444'}
    )
    fig_impact.update_layout(**plotly_layout_defaults)
    fig_impact.update_layout(yaxis=dict(range=[1, 5]), showlegend=False)

    # Chart 6: Review Score by State (State Bar Chart)
    state_reviews = rev_calc.reviews_by_state(df)
    fig_state = px.bar(
        state_reviews,
        x='customer_state',
        y='avg_review_score',
        labels={'customer_state': 'State', 'avg_review_score': 'Avg Rating'},
        color_discrete_sequence=['#06b6d4'] # Cyan
    )
    fig_state.update_layout(**plotly_layout_defaults)
    fig_state.update_layout(yaxis=dict(range=[1, 5]))

    graphs_json = {
        "score_dist": plotly_to_json(fig_dist),
        "sentiment_donut": plotly_to_json(fig_donut),
        "category_rating": plotly_to_json(fig_cat),
        "delivery_box": plotly_to_json(fig_box),
        "late_impact": plotly_to_json(fig_impact),
        "state_rating": plotly_to_json(fig_state)
    }

    # 4. Root Cause Analysis
    # A. Lowest Rated Categories
    worst_categories = rev_calc.low_rating_categories(df, min_reviews=10).head(5).to_dict('records')
    
    # B. Highest Delay Categories (Minimum 10 orders)
    category_delays = df.drop_duplicates(subset=['order_id']).groupby('product_category_name_english').agg(
        avg_delivery_days=('delivery_days', 'mean'),
        order_count=('delivery_days', 'count')
    ).reset_index()
    category_delays = category_delays[category_delays['order_count'] >= 10]
    worst_delay_categories = category_delays.sort_values(by='avg_delivery_days', ascending=False).head(5).to_dict('records')

    # C. Worst Performing States
    worst_states = state_reviews.sort_values(by='avg_review_score', ascending=True).head(5).to_dict('records')

    root_cause = {
        "worst_categories": worst_categories,
        "worst_delays": worst_delay_categories,
        "worst_states": worst_states
    }

    # 5. Insight Engine: Automated Satisfaction Findings
    insights = []
    
    # Review Score & Sentiment Insight
    insights.append({
        "type": "success" if avg_rating >= 4.0 else "warning",
        "icon": "bi-star-half",
        "title": "Overall Customer Satisfaction",
        "text": f"Olist maintains an average rating of <strong>{avg_rating:.2f} / 5.00</strong>, with <strong>{pos_rate:.1f}%</strong> of reviews classified as Positive (4-5 stars) and only <strong>{metrics['negative_rate']:.1f}%</strong> negative."
    })

    # Late Delivery Impact Insight
    rating_gap = ontime_avg - late_avg
    insights.append({
        "type": "danger" if rating_gap >= 1.0 else "warning",
        "icon": "bi-truck",
        "title": "Late Delivery Penalty",
        "text": f"Late deliveries experience a severe rating drop-off: On-Time orders average <strong>{ontime_avg:.2f} stars</strong>, whereas late orders average only <strong>{late_avg:.2f} stars</strong> (a satisfaction penalty of <strong>-{rating_gap:.2f} points</strong>)."
    })

    # Satisfaction by Retention Insight
    ret_gap = ret_sat["repeat_avg_rating"] - ret_sat["one_time_avg_rating"]
    insights.append({
        "type": "success" if ret_gap >= 0.1 else "info",
        "icon": "bi-people",
        "title": "Satisfaction vs Retention Link",
        "text": f"Repeat customers leave an average rating of <strong>{ret_sat['repeat_avg_rating']:.2f} stars</strong> compared to <strong>{ret_sat['one_time_avg_rating']:.2f} stars</strong> for one-time buyers. " +
               ("This indicates that returning customers have a higher satisfaction baseline." if ret_gap > 0 else "Low satisfaction directly limits customer conversion into repeat buyers.")
    })

    # Logistics bottleneck
    insights.append({
        "type": "danger" if metrics["late_delivery_rate"] >= 10.0 else "info",
        "icon": "bi-exclamation-circle",
        "title": "Logistics Delays Breakdown",
        "text": f"The late delivery rate stands at <strong>{metrics['late_delivery_rate']:.1f}%</strong>. When orders miss their SLA, they suffer an average delay of <strong>{metrics['avg_delay_days']:.1f} days</strong> beyond their estimated date."
    })

    # 6. Recommendation Engine
    lowest_cat_name = metrics["lowest_category"]
    lowest_cat_rating = metrics["lowest_category_rating"]
    
    recommendations = [
        {
            "finding": f"Late deliveries degrade review ratings by {rating_gap:.2f} stars on average.",
            "impact": "Courier delays are the leading driver of negative review clusters and customer churn.",
            "recommendation": "Set up automated push notifications and win-back coupons (e.g. $10 store credit) the moment an estimated delivery date is missed.",
            "priority": "Critical"
        },
        {
            "finding": f"Lowest rated category is '{lowest_cat_name}' with an average rating of {lowest_cat_rating:.2f} stars.",
            "impact": "Product quality issues are clusters of bad reviews, increasing refunds and service costs.",
            "recommendation": f"Perform a product quality audit on merchants listing items under '{lowest_cat_name}' and suspend those with ratings < 3.0.",
            "priority": "High"
        },
        {
            "finding": f"Worst performing state by ratings is {worst_states[0]['customer_state']} ({worst_states[0]['avg_review_score']:.2f} avg rating).",
            "impact": "Regional delivery bottlenecks in low-rating states degrade geographic customer lifetime value.",
            "recommendation": f"Review carrier transit times in {worst_states[0]['customer_state']} and onboard local fulfillment centers or alternate logistics networks.",
            "priority": "High"
        },
        {
            "finding": f"Satisfaction gap of {ret_gap:+.2f} stars between repeat ({ret_sat['repeat_avg_rating']:.2f}) and one-time ({ret_sat['one_time_avg_rating']:.2f}) buyers.",
            "impact": "Lower transactional ratings among first-time buyers block conversion to repeat cohorts.",
            "recommendation": "Establish an immediate post-purchase outreach sequence to one-time buyers who rate their first order < 3.0 to resolve issues and offer compensation.",
            "priority": "Medium"
        }
    ]

    kpis = {
        "avg_review_score": f"{avg_rating:.2f}",
        "positive_rate": f"{pos_rate:.1f}%",
        "negative_rate": f"{metrics['negative_rate']:.1f}%",
        "avg_delivery_days": f"{metrics['avg_delivery_days']:.1f}",
        "late_delivery_rate": f"{metrics['late_delivery_rate']:.1f}%",
        "on_time_delivery_rate": f"{ontime_rate:.1f}%",
        "lowest_category": lowest_cat_name,
        "lowest_category_rating": f"{lowest_cat_rating:.2f}"
    }

    return render_template(
        'reviews.html',
        active_page='reviews',
        loaded=True,
        kpis=kpis,
        ret_sat=ret_sat,
        health=health_indicators,
        graphs=graphs_json,
        root_cause=root_cause,
        insights=insights,
        recommendations=recommendations
    )

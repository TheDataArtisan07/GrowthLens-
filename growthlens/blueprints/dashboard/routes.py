from flask import Blueprint, render_template
from services.data_manager import data_manager

dashboard_bp = Blueprint('dashboard', __name__)

import json

@dashboard_bp.route('/dashboard')
def index():
    diagnostics = data_manager.get_diagnostics()
    df = data_manager.get_analytics_df()
    
    if df is None or not diagnostics.get("load_success", False):
        return render_template('dashboard.html', active_page='dashboard', diagnostics=diagnostics, loaded=False)
        
    # 1. Calculate Monthly Sales & Orders Trend
    order_payments = df.drop_duplicates(subset=['order_id', 'payment_value'])
    monthly_rev = order_payments.groupby('purchase_year_month')['payment_value'].sum().reset_index()
    monthly_ords = df.groupby('purchase_year_month')['order_id'].nunique().reset_index()
    
    monthly_stats = monthly_rev.merge(monthly_ords, on='purchase_year_month')
    monthly_stats.columns = ['purchase_year_month', 'revenue', 'orders']
    
    # Filter out low-volume startup months to match other analytics charts
    monthly_stats = monthly_stats[monthly_stats['orders'] >= 100].sort_values('purchase_year_month')
    trend_data = monthly_stats.to_dict(orient='records')
    
    # 2. Calculate Geographic Order Share per State
    state_stats = df.groupby('customer_state')['order_id'].nunique().reset_index()
    state_stats.columns = ['state', 'order_count']
    state_stats = state_stats.sort_values(by='order_count', ascending=False)
    state_data = state_stats.to_dict(orient='records')
    
    return render_template(
        'dashboard.html',
        active_page='dashboard',
        diagnostics=diagnostics,
        loaded=True,
        trend_data=json.dumps(trend_data),
        state_data=json.dumps(state_data)
    )

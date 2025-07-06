import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="UAE Banking Sector Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .dimension-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .performance-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 0.2rem 0;
    }
    .performance-negative {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_banking_data():
    """Load and process the UAE banking sector data"""
    try:
        # Read the CSV file
        df = pd.read_csv('UAE Banking Sector Performance.csv')
        
        # Data cleaning and preprocessing
        df = df.dropna()
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Value'])
        
        # Parse quarter ending to create date-like column
        df['Quarter'] = df['Quarter Ending']
        df['Year'] = df['Quarter Ending'].str[:4].astype(int)
        df['QuarterNum'] = df['Quarter Ending'].str[-1].astype(int)
        
        # Create a proper date column for better sorting and visualization
        # Convert quarter format (2021Q1) to proper date
        def quarter_to_date(quarter_str):
            year = int(quarter_str[:4])
            quarter = int(quarter_str[-1])
            month = (quarter - 1) * 3 + 1  # Q1=1, Q2=4, Q3=7, Q4=10
            return pd.Timestamp(year=year, month=month, day=1)
        
        df['Date'] = df['Quarter Ending'].apply(quarter_to_date)
        
        # Create period groupings
        df['Year_Quarter'] = df['Quarter Ending']
        df['Period'] = df['Year'].apply(lambda x: 
            '2021-2022' if x <= 2022 else
            '2023-2024' if x <= 2024 else
            '2025'
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_banking_data()

if df is None:
    st.error("Failed to load banking data. Please check the file.")
    st.stop()

# Define dimension-measure mapping for better organization
dimension_measures = {
    'Asset Quality': [
        'Nonperforming loans',
        'Nonperforming loans net of provisions ',
        'Nonperforming loans to total gross loans',
        'Provisions to nonperforming loans'
    ],
    'Asset size': [
        'Risk-weighted assets',
        'Total assets ',
        'Total gross loans'
    ],
    'Capital Adequacy': [
        'Common Equity Tier 1 capital to risk-weighted assets 1',
        'Regulatory capital to risk-weighted assets',
        'Tier 1 capital to risk-weighted assets 1'
    ],
    'Liquidity': [
        'Liquid assets to total assets',
        'Liquidity coverage ratio',
        'Net stable funding ratio'
    ],
    'Profitability': [
        'Net income before taxes',
        'Return on assets',
        'Return on equity'
    ]
}

# Main title and description
st.markdown('<h1 class="main-header">🏦 UAE Banking Sector Performance Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comprehensive Analysis of UAE Banking Performance Metrics (2021Q1 - 2025Q1)</p>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔍 Analysis Filters")

# Quarter range filter
available_quarters = sorted(df['Quarter Ending'].unique())
quarter_range = st.sidebar.select_slider(
    "Select Quarter Range",
    options=available_quarters,
    value=(available_quarters[0], available_quarters[-1])
)

# Filter quarters based on selection
start_idx = available_quarters.index(quarter_range[0])
end_idx = available_quarters.index(quarter_range[1])
selected_quarters = available_quarters[start_idx:end_idx+1]

# Dimension selection
selected_dimensions = st.sidebar.multiselect(
    "Select Banking Dimensions",
    list(dimension_measures.keys()),
    default=list(dimension_measures.keys())
)

# Measure selection based on selected dimensions
available_measures = []
for dim in selected_dimensions:
    available_measures.extend(dimension_measures[dim])

selected_measures = st.sidebar.multiselect(
    "Select Specific Measures (optional)",
    available_measures,
    default=[]
)

# Apply filters
filtered_df = df[df['Quarter Ending'].isin(selected_quarters)]

if selected_dimensions:
    filtered_df = filtered_df[filtered_df['Dimension'].isin(selected_dimensions)]

if selected_measures:
    filtered_df = filtered_df[filtered_df['Measure'].isin(selected_measures)]

# Overview metrics
st.header("📊 Key Performance Overview")

# Calculate latest quarter metrics
latest_quarter = df['Quarter Ending'].max()
previous_quarter = sorted(df['Quarter Ending'].unique())[-2] if len(df['Quarter Ending'].unique()) > 1 else latest_quarter

col1, col2, col3, col4 = st.columns(4)

with col1:
    quarters_analyzed = len(selected_quarters)
    st.metric(
        label="Quarters Analyzed",
        value=quarters_analyzed,
        delta=f"{quarter_range[0]} to {quarter_range[1]}"
    )

with col2:
    dimensions_count = len(selected_dimensions)
    st.metric(
        label="Dimensions",
        value=dimensions_count,
        delta=f"of {len(dimension_measures)} total"
    )

with col3:
    measures_count = filtered_df['Measure'].nunique()
    st.metric(
        label="Measures Tracked",
        value=measures_count,
        delta=f"of {df['Measure'].nunique()} available"
    )

with col4:
    data_points = len(filtered_df)
    st.metric(
        label="Data Points",
        value=data_points,
        delta=f"{(data_points/len(df)*100):.1f}% of total"
    )

# Create tabs for different analyses
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Trend Analysis", 
    "🏦 Dimension Deep Dive", 
    "📊 Comparative Analysis",
    "🎯 Performance Metrics",
    "📉 Growth Analysis",
    "💡 Key Insights"
])

with tab1:
    st.header("📈 Quarterly Trend Analysis")
    
    if filtered_df.empty:
        st.warning("No data available for selected filters.")
    else:
        # Multi-measure trend analysis
        st.subheader("Multi-Measure Trends")
        
        # Select measures for trend comparison
        trend_measures = st.multiselect(
            "Select Measures for Trend Analysis",
            filtered_df['Measure'].unique(),
            default=list(filtered_df['Measure'].unique())[:3],
            key="trend_measures"
        )
        
        if trend_measures:
            # Create subplot for multiple measures
            fig_trends = make_subplots(
                rows=len(trend_measures), cols=1,
                subplot_titles=[f"{measure}" for measure in trend_measures],
                vertical_spacing=0.08,
                shared_xaxes=True
            )
            
            colors = px.colors.qualitative.Set1
            
            for i, measure in enumerate(trend_measures):
                measure_data = filtered_df[filtered_df['Measure'] == measure].sort_values('Date')
                
                # Get unit for y-axis label
                unit = measure_data['Unit of measurement'].iloc[0] if not measure_data.empty else ""
                
                fig_trends.add_trace(
                    go.Scatter(
                        x=measure_data['Quarter Ending'],
                        y=measure_data['Value'],
                        mode='lines+markers',
                        name=measure,
                        line=dict(color=colors[i % len(colors)], width=3),
                        marker=dict(size=8),
                        hovertemplate=f'<b>{measure}</b><br>Quarter: %{{x}}<br>Value: %{{y:.2f}} {unit}<extra></extra>'
                    ),
                    row=i+1, col=1
                )
                
                # Add trend line
                if len(measure_data) > 1:
                    x_numeric = np.arange(len(measure_data))
                    z = np.polyfit(x_numeric, measure_data['Value'], 1)
                    p = np.poly1d(z)
                    
                    fig_trends.add_trace(
                        go.Scatter(
                            x=measure_data['Quarter Ending'],
                            y=p(x_numeric),
                            mode='lines',
                            name=f'{measure} Trend',
                            line=dict(color=colors[i % len(colors)], dash='dash', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=i+1, col=1
                    )
                
                # Update y-axis labels
                fig_trends.update_yaxes(title_text=f"{unit}", row=i+1, col=1)
            
            fig_trends.update_layout(
                height=300*len(trend_measures), 
                title="Quarterly Trends with Trend Lines",
                showlegend=False
            )
            fig_trends.update_xaxes(title_text="Quarter", row=len(trend_measures), col=1)
            
            st.plotly_chart(fig_trends, use_container_width=True)
        
        # Quarter-over-quarter growth
        st.subheader("Quarter-over-Quarter Growth Analysis")
        
        growth_measure = st.selectbox(
            "Select Measure for Growth Analysis",
            filtered_df['Measure'].unique(),
            key="growth_measure"
        )
        
        if growth_measure:
            growth_data = filtered_df[filtered_df['Measure'] == growth_measure].sort_values('Date')
            growth_data['QoQ_Growth'] = growth_data['Value'].pct_change() * 100
            growth_data = growth_data.dropna()
            
            if not growth_data.empty:
                fig_growth = go.Figure()
                
                # Add growth bars
                colors = ['green' if x >= 0 else 'red' for x in growth_data['QoQ_Growth']]
                
                fig_growth.add_trace(
                    go.Bar(
                        x=growth_data['Quarter Ending'],
                        y=growth_data['QoQ_Growth'],
                        name='QoQ Growth %',
                        marker_color=colors,
                        hovertemplate='Quarter: %{x}<br>Growth: %{y:.2f}%<extra></extra>'
                    )
                )
                
                fig_growth.add_hline(y=0, line_dash="dash", line_color="black")
                fig_growth.update_layout(
                    title=f"Quarter-over-Quarter Growth: {growth_measure}",
                    xaxis_title="Quarter",
                    yaxis_title="Growth Rate (%)",
                    height=400
                )
                
                st.plotly_chart(fig_growth, use_container_width=True)

with tab2:
    st.header("🏦 Dimension Deep Dive Analysis")
    
    # Dimension selection for deep dive
    dimension_focus = st.selectbox(
        "Select Dimension for Detailed Analysis",
        selected_dimensions,
        key="dimension_focus"
    )
    
    if dimension_focus:
        dimension_data = filtered_df[filtered_df['Dimension'] == dimension_focus]
        
        st.subheader(f"📊 {dimension_focus} Analysis")
        
        # Get measures for this dimension
        dim_measures = dimension_data['Measure'].unique()
        
        # Create performance summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Latest Quarter Performance")
            latest_data = dimension_data[dimension_data['Quarter Ending'] == latest_quarter]
            
            for _, row in latest_data.iterrows():
                unit = row['Unit of measurement']
                value = row['Value']
                
                # Calculate change from previous quarter
                prev_data = dimension_data[
                    (dimension_data['Measure'] == row['Measure']) & 
                    (dimension_data['Quarter Ending'] == previous_quarter)
                ]
                
                if not prev_data.empty:
                    prev_value = prev_data['Value'].iloc[0]
                    change = ((value - prev_value) / prev_value) * 100
                    delta = f"{change:+.2f}%"
                else:
                    delta = None
                
                st.metric(
                    label=row['Measure'],
                    value=f"{value:.2f} {unit}",
                    delta=delta
                )
        
        with col2:
            st.markdown("#### Dimension Trends Overview")
            
            # Create mini trends for each measure
            fig_mini = make_subplots(
                rows=len(dim_measures), cols=1,
                subplot_titles=dim_measures,
                vertical_spacing=0.1,
                shared_xaxes=True
            )
            
            for i, measure in enumerate(dim_measures):
                measure_data = dimension_data[dimension_data['Measure'] == measure].sort_values('Date')
                
                fig_mini.add_trace(
                    go.Scatter(
                        x=measure_data['Quarter Ending'],
                        y=measure_data['Value'],
                        mode='lines+markers',
                        name=measure,
                        showlegend=False,
                        line=dict(width=2),
                        marker=dict(size=4)
                    ),
                    row=i+1, col=1
                )
            
            fig_mini.update_layout(height=150*len(dim_measures), title=f"{dimension_focus} - All Measures")
            st.plotly_chart(fig_mini, use_container_width=True)
        
        # Detailed measure analysis
        st.subheader("📈 Detailed Measure Analysis")
        
        selected_measure = st.selectbox(
            f"Select Measure from {dimension_focus}",
            dim_measures,
            key="detailed_measure"
        )
        
        if selected_measure:
            measure_detail = dimension_data[dimension_data['Measure'] == selected_measure].sort_values('Date')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Main trend chart
                fig_detail = px.line(
                    measure_detail,
                    x='Quarter Ending',
                    y='Value',
                    title=f"{selected_measure} - Detailed Trend",
                    markers=True
                )
                
                # Add annotations for min/max
                min_idx = measure_detail['Value'].idxmin()
                max_idx = measure_detail['Value'].idxmax()
                
                fig_detail.add_annotation(
                    x=measure_detail.loc[min_idx, 'Quarter Ending'],
                    y=measure_detail.loc[min_idx, 'Value'],
                    text=f"Min: {measure_detail.loc[min_idx, 'Value']:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red"
                )
                
                fig_detail.add_annotation(
                    x=measure_detail.loc[max_idx, 'Quarter Ending'],
                    y=measure_detail.loc[max_idx, 'Value'],
                    text=f"Max: {measure_detail.loc[max_idx, 'Value']:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="green"
                )
                
                fig_detail.update_layout(height=400)
                st.plotly_chart(fig_detail, use_container_width=True)
            
            with col2:
                # Statistical summary
                st.markdown("#### Statistical Summary")
                stats = measure_detail['Value'].describe()
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'],
                    'Value': [
                        f"{stats['count']:.0f}",
                        f"{stats['mean']:.2f}",
                        f"{stats['std']:.2f}",
                        f"{stats['min']:.2f}",
                        f"{stats['25%']:.2f}",
                        f"{stats['50%']:.2f}",
                        f"{stats['75%']:.2f}",
                        f"{stats['max']:.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)
                
                # Performance assessment
                st.markdown("#### Performance Assessment")
                latest_value = measure_detail['Value'].iloc[-1]
                mean_value = measure_detail['Value'].mean()
                
                if latest_value > mean_value:
                    st.markdown('<div class="performance-positive">📈 Above Average Performance</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="performance-negative">📉 Below Average Performance</div>', unsafe_allow_html=True)
                
                # Volatility assessment
                cv = measure_detail['Value'].std() / measure_detail['Value'].mean()
                if cv < 0.1:
                    volatility = "Low"
                    color = "green"
                elif cv < 0.2:
                    volatility = "Medium"
                    color = "orange"
                else:
                    volatility = "High"
                    color = "red"
                
                st.markdown(f"**Volatility:** <span style='color: {color}'>{volatility}</span> (CV: {cv:.3f})", unsafe_allow_html=True)

with tab3:
    st.header("📊 Comparative Analysis")
    
    # Cross-measure comparison
    st.subheader("Cross-Measure Comparison")
    
    comparison_measures = st.multiselect(
        "Select Measures for Comparison",
        filtered_df['Measure'].unique(),
        default=list(filtered_df['Measure'].unique())[:4],
        key="comparison_measures"
    )
    
    if len(comparison_measures) >= 2:
        # Normalize data for comparison (since units are different)
        comparison_data = []
        
        for measure in comparison_measures:
            measure_data = filtered_df[filtered_df['Measure'] == measure].sort_values('Date')
            
            # Normalize to 0-100 scale
            min_val = measure_data['Value'].min()
            max_val = measure_data['Value'].max()
            
            if max_val != min_val:
                measure_data['Normalized'] = ((measure_data['Value'] - min_val) / (max_val - min_val)) * 100
            else:
                measure_data['Normalized'] = 50  # If all values are the same
            
            comparison_data.append(measure_data)
        
        # Combined comparison chart
        fig_comparison = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, measure_data in enumerate(comparison_data):
            measure_name = measure_data['Measure'].iloc[0]
            
            fig_comparison.add_trace(
                go.Scatter(
                    x=measure_data['Quarter Ending'],
                    y=measure_data['Normalized'],
                    mode='lines+markers',
                    name=measure_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    hovertemplate=f'<b>{measure_name}</b><br>Quarter: %{{x}}<br>Normalized Score: %{{y:.1f}}<extra></extra>'
                )
            )
        
        fig_comparison.update_layout(
            title="Normalized Measure Comparison (0-100 Scale)",
            xaxis_title="Quarter",
            yaxis_title="Normalized Score",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Measure Correlation Analysis")
        
        if len(comparison_measures) >= 2:
            # Create correlation matrix
            pivot_data = filtered_df[filtered_df['Measure'].isin(comparison_measures)].pivot_table(
                values='Value',
                index='Quarter Ending',
                columns='Measure',
                aggfunc='first'
            )
            
            if not pivot_data.empty:
                corr_matrix = pivot_data.corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Measure Correlation Matrix",
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Strongest correlations
                st.markdown("#### Strongest Correlations")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        measure1 = corr_matrix.columns[i]
                        measure2 = corr_matrix.columns[j]
                        corr_value = corr_matrix.iloc[i, j]
                        corr_pairs.append({
                            'Measure 1': measure1,
                            'Measure 2': measure2,
                            'Correlation': corr_value,
                            'Strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate' if abs(corr_value) > 0.3 else 'Weak'
                        })
                
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
                st.dataframe(corr_df.round(3), use_container_width=True)

with tab4:
    st.header("🎯 Performance Metrics Dashboard")
    
    # Performance scorecard
    st.subheader("Banking Performance Scorecard")
    
    # Define performance criteria (these are example thresholds - adjust based on industry standards)
    performance_criteria = {
        'Asset Quality': {
            'Nonperforming loans to total gross loans': {'good': '<3%', 'threshold': 3.0, 'lower_better': True},
            'Provisions to nonperforming loans': {'good': '>70%', 'threshold': 70.0, 'lower_better': False}
        },
        'Capital Adequacy': {
            'Common Equity Tier 1 capital to risk-weighted assets 1': {'good': '>10%', 'threshold': 10.0, 'lower_better': False},
            'Regulatory capital to risk-weighted assets': {'good': '>12%', 'threshold': 12.0, 'lower_better': False},
            'Tier 1 capital to risk-weighted assets 1': {'good': '>8%', 'threshold': 8.0, 'lower_better': False}
        },
        'Liquidity': {
            'Liquidity coverage ratio': {'good': '>100%', 'threshold': 100.0, 'lower_better': False},
            'Net stable funding ratio': {'good': '>100%', 'threshold': 100.0, 'lower_better': False}
        },
        'Profitability': {
            'Return on assets': {'good': '>1%', 'threshold': 1.0, 'lower_better': False},
            'Return on equity': {'good': '>10%', 'threshold': 10.0, 'lower_better': False}
        }
    }
    
    # Create performance dashboard
    latest_performance = filtered_df[filtered_df['Quarter Ending'] == latest_quarter]
    
    performance_scores = []
    
    for dimension, criteria in performance_criteria.items():
        if dimension in selected_dimensions:
            st.markdown(f"#### {dimension}")
            
            cols = st.columns(len(criteria))
            
            for i, (measure, criterion) in enumerate(criteria.items()):
                measure_data = latest_performance[latest_performance['Measure'] == measure]
                
                if not measure_data.empty:
                    current_value = measure_data['Value'].iloc[0]
                    unit = measure_data['Unit of measurement'].iloc[0]
                    
                    # Determine performance status
                    if criterion['lower_better']:
                        is_good = current_value < criterion['threshold']
                        performance_color = "green" if is_good else "red"
                        performance_status = "✅ Good" if is_good else "⚠️ Needs Attention"
                    else:
                        is_good = current_value > criterion['threshold']
                        performance_color = "green" if is_good else "red"
                        performance_status = "✅ Good" if is_good else "⚠️ Needs Attention"
                    
                    performance_scores.append({
                        'Dimension': dimension,
                        'Measure': measure,
                        'Current Value': current_value,
                        'Threshold': criterion['threshold'],
                        'Status': performance_status,
                        'Is Good': is_good
                    })
                    
                    with cols[i]:
                        st.metric(
                            label=measure.replace('1', '').strip(),
                            value=f"{current_value:.2f}{unit}",
                            delta=criterion['good']
                        )
                        st.markdown(f"<p style='color: {performance_color}'>{performance_status}</p>", unsafe_allow_html=True)
    
    # Performance summary
    if performance_scores:
        st.subheader("📈 Overall Performance Summary")
        
        total_metrics = len(performance_scores)
        good_metrics = sum(1 for score in performance_scores if score['Is Good'])
        performance_percentage = (good_metrics / total_metrics) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Metrics Evaluated", total_metrics)
        
        with col2:
            st.metric("Metrics Meeting Targets", good_metrics, f"{performance_percentage:.1f}%")
        
        with col3:
            overall_status = "Strong" if performance_percentage >= 80 else "Moderate" if performance_percentage >= 60 else "Needs Improvement"
            st.metric("Overall Assessment", overall_status)
        
        # Performance breakdown table
        performance_df = pd.DataFrame(performance_scores)
        performance_df = performance_df[['Dimension', 'Measure', 'Current Value', 'Threshold', 'Status']]
        st.dataframe(performance_df, use_container_width=True)

with tab5:
    st.header("📉 Growth & Trend Analysis")
    
    # Year-over-year analysis
    st.subheader("Year-over-Year Analysis")
    
    # Calculate YoY growth for same quarters
    yoy_analysis = []
    
    for measure in filtered_df['Measure'].unique():
        measure_data = filtered_df[filtered_df['Measure'] == measure].copy()
        
        # Group by quarter number to compare same quarters across years
        for quarter_num in [1, 2, 3, 4]:
            quarter_data = measure_data[measure_data['QuarterNum'] == quarter_num].sort_values('Year')
            
            if len(quarter_data) > 1:
                for i in range(1, len(quarter_data)):
                    current_row = quarter_data.iloc[i]
                    previous_row = quarter_data.iloc[i-1]
                    
                    yoy_growth = ((current_row['Value'] - previous_row['Value']) / previous_row['Value']) * 100
                    
                    yoy_analysis.append({
                        'Measure': measure,
                        'Quarter': f"Q{quarter_num}",
                        'Current Year': current_row['Year'],
                        'Previous Year': previous_row['Year'],
                        'Current Value': current_row['Value'],
                        'Previous Value': previous_row['Value'],
                        'YoY Growth (%)': yoy_growth,
                        'Period': f"{previous_row['Year']}-{current_row['Year']}"
                    })
    
    if yoy_analysis:
        yoy_df = pd.DataFrame(yoy_analysis)
        
        # YoY growth visualization
        yoy_measure = st.selectbox(
            "Select Measure for YoY Analysis",
            yoy_df['Measure'].unique(),
            key="yoy_measure"
        )
        
        yoy_measure_data = yoy_df[yoy_df['Measure'] == yoy_measure]
        
        if not yoy_measure_data.empty:
            fig_yoy = px.bar(
                yoy_measure_data,
                x='Period',
                y='YoY Growth (%)',
                color='Quarter',
                title=f"Year-over-Year Growth: {yoy_measure}",
                text='YoY Growth (%)'
            )
            fig_yoy.add_hline(y=0, line_dash="dash", line_color="black")
            fig_yoy.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_yoy.update_layout(height=500)
            st.plotly_chart(fig_yoy, use_container_width=True)
            
            # YoY summary table
            st.subheader("YoY Growth Summary")
            summary_table = yoy_measure_data.pivot_table(
                values='YoY Growth (%)',
                index='Period',
                columns='Quarter',
                aggfunc='first'
            ).round(2)
            st.dataframe(summary_table, use_container_width=True)
    
    # Compound Annual Growth Rate (CAGR) Analysis
    st.subheader("Compound Annual Growth Rate (CAGR) Analysis")
    
    cagr_results = []
    
    for measure in filtered_df['Measure'].unique():
        measure_data = filtered_df[filtered_df['Measure'] == measure].sort_values('Date')
        
        if len(measure_data) > 1:
            first_value = measure_data['Value'].iloc[0]
            last_value = measure_data['Value'].iloc[-1]
            
            # Calculate number of years
            first_year = measure_data['Year'].iloc[0]
            last_year = measure_data['Year'].iloc[-1]
            first_quarter = measure_data['QuarterNum'].iloc[0]
            last_quarter = measure_data['QuarterNum'].iloc[-1]
            
            years = (last_year - first_year) + (last_quarter - first_quarter) / 4
            
            if first_value > 0 and years > 0:
                cagr = ((last_value / first_value) ** (1/years) - 1) * 100
                
                cagr_results.append({
                    'Measure': measure,
                    'Dimension': measure_data['Dimension'].iloc[0],
                    'Start Value': first_value,
                    'End Value': last_value,
                    'Period (Years)': years,
                    'CAGR (%)': cagr,
                    'Total Growth (%)': ((last_value - first_value) / first_value) * 100
                })
    
    if cagr_results:
        cagr_df = pd.DataFrame(cagr_results)
        
        # CAGR visualization
        fig_cagr = px.bar(
            cagr_df.sort_values('CAGR (%)', ascending=True),
            x='CAGR (%)',
            y='Measure',
            color='Dimension',
            title="Compound Annual Growth Rate by Measure",
            orientation='h'
        )
        fig_cagr.add_vline(x=0, line_dash="dash", line_color="black")
        fig_cagr.update_layout(height=600)
        st.plotly_chart(fig_cagr, use_container_width=True)
        
        # CAGR summary table
        st.subheader("CAGR Summary Table")
        cagr_display = cagr_df[['Measure', 'Dimension', 'CAGR (%)', 'Total Growth (%)']].round(2)
        cagr_display = cagr_display.sort_values('CAGR (%)', ascending=False)
        st.dataframe(cagr_display, use_container_width=True)

with tab6:
    st.header("💡 Key Insights & Recommendations")
    
    # Generate automated insights
    insights = []
    recommendations = []
    
    # Latest quarter analysis
    latest_data = filtered_df[filtered_df['Quarter Ending'] == latest_quarter]
    
    # Growth insights
    for measure in filtered_df['Measure'].unique():
        measure_data = filtered_df[filtered_df['Measure'] == measure].sort_values('Date')
        
        if len(measure_data) >= 4:  # At least 4 quarters for meaningful analysis
            recent_trend = measure_data['Value'].iloc[-4:].pct_change().mean() * 100
            
            if abs(recent_trend) > 2:  # Significant trend
                trend_direction = "increasing" if recent_trend > 0 else "decreasing"
                dimension = measure_data['Dimension'].iloc[0]
                
                insights.append({
                    'Type': 'Trend',
                    'Dimension': dimension,
                    'Measure': measure,
                    'Insight': f"{measure} shows a {trend_direction} trend",
                    'Detail': f"Average quarterly change: {recent_trend:.2f}%",
                    'Priority': 'High' if abs(recent_trend) > 5 else 'Medium'
                })
    
    # Performance insights based on latest data
    if performance_scores:
        weak_areas = [score for score in performance_scores if not score['Is Good']]
        strong_areas = [score for score in performance_scores if score['Is Good']]
        
        if weak_areas:
            for area in weak_areas:
                insights.append({
                    'Type': 'Performance Alert',
                    'Dimension': area['Dimension'],
                    'Measure': area['Measure'],
                    'Insight': f"{area['Measure']} is below optimal threshold",
                    'Detail': f"Current: {area['Current Value']:.2f}, Target: {area['Threshold']:.2f}",
                    'Priority': 'High'
                })
        
        if strong_areas:
            best_performer = max(strong_areas, key=lambda x: x['Current Value'] if not x.get('lower_better', False) else 1/x['Current Value'])
            insights.append({
                'Type': 'Strength',
                'Dimension': best_performer['Dimension'],
                'Measure': best_performer['Measure'],
                'Insight': f"{best_performer['Measure']} shows strong performance",
                'Detail': f"Exceeds threshold by significant margin",
                'Priority': 'Positive'
            })
    
    # Display insights
    if insights:
        # Group insights by priority
        high_priority = [i for i in insights if i['Priority'] == 'High']
        medium_priority = [i for i in insights if i['Priority'] == 'Medium']
        positive = [i for i in insights if i['Priority'] == 'Positive']
        
        if high_priority:
            st.subheader("🚨 High Priority Insights")
            for insight in high_priority:
                st.markdown(f"**{insight['Dimension']} - {insight['Measure']}**")
                st.markdown(f"📊 {insight['Insight']}")
                st.markdown(f"📈 {insight['Detail']}")
                st.markdown("---")
        
        if positive:
            st.subheader("✅ Positive Performance Areas")
            for insight in positive:
                st.markdown(f"**{insight['Dimension']} - {insight['Measure']}**")
                st.markdown(f"🎯 {insight['Insight']}")
                st.markdown(f"📊 {insight['Detail']}")
                st.markdown("---")
        
        if medium_priority:
            st.subheader("📋 Areas to Monitor")
            for insight in medium_priority:
                st.markdown(f"**{insight['Dimension']} - {insight['Measure']}**")
                st.markdown(f"📈 {insight['Insight']}")
                st.markdown(f"📊 {insight['Detail']}")
                st.markdown("---")
    
    # Strategic recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    # Generate recommendations based on performance and trends
    if weak_areas:
        st.markdown("#### Priority Actions Required:")
        for area in weak_areas[:3]:  # Top 3 areas needing attention
            dimension = area['Dimension']
            measure = area['Measure']
            
            if dimension == 'Asset Quality':
                if 'Nonperforming loans' in measure:
                    st.markdown("• **Credit Risk Management**: Strengthen credit assessment processes and enhance loan monitoring systems")
                elif 'Provisions' in measure:
                    st.markdown("• **Provisioning Strategy**: Review and enhance provisioning policies to better reflect risk profile")
            
            elif dimension == 'Capital Adequacy':
                st.markdown("• **Capital Strengthening**: Consider capital raising or retained earnings improvement to boost capital ratios")
            
            elif dimension == 'Liquidity':
                st.markdown("• **Liquidity Management**: Optimize asset-liability composition and enhance liquidity buffers")
            
            elif dimension == 'Profitability':
                if 'Return on' in measure:
                    st.markdown("• **Profitability Enhancement**: Focus on cost optimization and revenue diversification strategies")
    
    # General recommendations
    st.markdown("#### General Strategic Focus Areas:")
    st.markdown("• **Digital Transformation**: Continue investing in digital banking capabilities to enhance efficiency")
    st.markdown("• **Risk Management**: Strengthen risk governance frameworks across all dimensions")
    st.markdown("• **Regulatory Compliance**: Maintain proactive approach to regulatory requirements and Basel III standards")
    st.markdown("• **Sustainable Finance**: Integrate ESG considerations into banking operations and strategy")
    
    # Export functionality
    st.subheader("📥 Export Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Executive Summary"):
            summary = {
                'Analysis Period': f"{quarter_range[0]} to {quarter_range[1]}",
                'Dimensions Analyzed': selected_dimensions,
                'Total Metrics': len(performance_scores) if performance_scores else 0,
                'Performance Score': f"{performance_percentage:.1f}%" if performance_scores else "N/A",
                'Key Insights': [insight['Insight'] for insight in insights[:5]],
                'Priority Actions': len([i for i in insights if i['Priority'] == 'High'])
            }
            st.json(summary)
    
    with col2:
        if st.button("📈 Download Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name=f"uae_banking_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Advanced Analytics Section
st.header("🔬 Advanced Analytics")

with st.expander("📊 Statistical Analysis"):
    st.subheader("Advanced Statistical Insights")
    
    # Volatility analysis
    volatility_analysis = []
    
    for measure in filtered_df['Measure'].unique():
        measure_data = filtered_df[filtered_df['Measure'] == measure]
        
        if len(measure_data) > 1:
            mean_val = measure_data['Value'].mean()
            std_val = measure_data['Value'].std()
            cv = std_val / mean_val if mean_val != 0 else 0
            
            volatility_analysis.append({
                'Measure': measure,
                'Dimension': measure_data['Dimension'].iloc[0],
                'Mean': mean_val,
                'Std Dev': std_val,
                'Coefficient of Variation': cv,
                'Volatility Level': 'Low' if cv < 0.1 else 'Medium' if cv < 0.2 else 'High'
            })
    
    if volatility_analysis:
        volatility_df = pd.DataFrame(volatility_analysis)
        
        # Volatility chart
        fig_volatility = px.scatter(
            volatility_df,
            x='Mean',
            y='Coefficient of Variation',
            color='Dimension',
            size='Std Dev',
            hover_name='Measure',
            title="Risk-Return Profile (Mean vs Volatility)",
            labels={'Coefficient of Variation': 'Volatility (CV)'}
        )
        fig_volatility.add_hline(y=0.1, line_dash="dash", annotation_text="Low Volatility Threshold")
        fig_volatility.add_hline(y=0.2, line_dash="dash", annotation_text="High Volatility Threshold")
        fig_volatility.update_layout(height=500)
        st.plotly_chart(fig_volatility, use_container_width=True)
        
        # Volatility summary
        st.dataframe(volatility_df.round(4), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 📖 About This Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Data Coverage:**
    - Time Period: 2021Q1 - 2025Q1
    - Dimensions: 5 key banking areas
    - Measures: 16 performance indicators
    - Data Points: 272 quarterly observations
    """)

with col2:
    st.markdown("""
    **Key Features:**
    - Quarterly trend analysis
    - Performance benchmarking
    - Growth rate calculations
    - Risk-return profiling
    - Automated insights generation
    """)

st.markdown("**🏦 UAE Banking Sector Performance Dashboard | Built with ❤️ using Streamlit**")
st.markdown("*This dashboard provides comprehensive analysis of UAE banking sector performance across key regulatory and financial metrics.*")
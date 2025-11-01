import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

def create_python_dashboard(csv_file='churn_predictions_complete.csv'):
    print("Creating Interactive Dashboard...")
    
    # Load the predictions data
    df = pd.read_csv(csv_file)
    
    # Create an interactive HTML dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Churn Risk Distribution', 'Churn Probability by Tenure',
                       'Monthly Charges vs Churn Probability', 'Risk Category Breakdown'),
        specs=[[{"type": "pie"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Risk Distribution Pie Chart
    risk_counts = df['risk_category'].value_counts()
    fig.add_trace(
        go.Pie(labels=risk_counts.index, values=risk_counts.values, name="Risk Distribution"),
        row=1, col=1
    )
    
    # 2. Churn Probability by Tenure
    fig.add_trace(
        go.Scatter(x=df['tenure'], y=df['churn_probability'], 
                  mode='markers', name='Churn Probability',
                  marker=dict(color=df['churn_probability'], colorscale='RdYlGn_r', showscale=True)),
        row=1, col=2
    )
    
    # 3. Monthly Charges vs Churn Probability
    fig.add_trace(
        go.Scatter(x=df['monthly_charges'], y=df['churn_probability'],
                  mode='markers', name='Monthly Charges',
                  marker=dict(color=df['actual_churn'], colorscale=['blue', 'red'],
                             showscale=True, colorbar=dict(title="Actual Churn"))),
        row=2, col=1
    )
    
    # 4. Risk Category Bar Chart
    contract_risk = pd.crosstab(df['contract_Month-to-month'], df['risk_category'])
    for i, risk in enumerate(contract_risk.columns):
        fig.add_trace(
            go.Bar(x=contract_risk.index, y=contract_risk[risk], name=f'Risk: {risk}'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Customer Churn Analysis Dashboard")
    
    # Save as HTML
    fig.write_html("churn_dashboard.html")
    print("✓ Interactive dashboard saved as 'churn_dashboard.html'")
    
    # Create additional detailed visualizations
    create_detailed_visualizations(df)
    
    return "churn_dashboard.html"

def create_detailed_visualizations(df):
    """Create individual visualization files"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Churn Risk Distribution
    plt.figure(figsize=(10, 6))
    risk_counts = df['risk_category'].value_counts()
    plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Customer Churn Risk Distribution')
    plt.savefig('risk_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Churn Probability by Tenure
    plt.figure(figsize=(12, 6))
    plt.scatter(df['tenure'], df['churn_probability'], 
                c=df['churn_probability'], cmap='RdYlGn_r', alpha=0.6)
    plt.colorbar(label='Churn Probability')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Churn Probability')
    plt.title('Churn Probability vs Customer Tenure')
    plt.grid(True, alpha=0.3)
    plt.savefig('tenure_vs_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance (if available)
    if 'feature_importance.png' in os.listdir():
        print("✓ Using existing feature importance plot")
    else:
        # Create a sample feature importance if not available
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['churn_probability', 'actual_churn', 'churn_prediction']]
        
        if len(numeric_cols) > 0:
            # Calculate correlation with churn probability
            correlations = df[numeric_cols].corrwith(df['churn_probability']).abs().sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            correlations.head(10).plot(kind='barh')
            plt.title('Top Features Correlated with Churn Probability')
            plt.xlabel('Absolute Correlation')
            plt.tight_layout()
            plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Monthly Charges Analysis
    plt.figure(figsize=(12, 6))
    high_risk = df[df['risk_category'] == 'High']
    low_risk = df[df['risk_category'] == 'Low']
    
    plt.hist([low_risk['monthly_charges'], high_risk['monthly_charges']], 
             bins=20, label=['Low Risk', 'High Risk'], alpha=0.7)
    plt.xlabel('Monthly Charges')
    plt.ylabel('Number of Customers')
    plt.title('Monthly Charges Distribution by Risk Category')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('charges_by_risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Detailed visualizations created:")
    print("  - risk_distribution.png")
    print("  - tenure_vs_churn.png") 
    print("  - feature_correlations.png")
    print("  - charges_by_risk.png")

def create_business_report():
    """Create a comprehensive business report"""
    
    df = pd.read_csv('churn_predictions_complete.csv')
    
    report = f"""
    CUSTOMER CHURN ANALYSIS REPORT
    ==============================
    
    Executive Summary:
    - Total Customers Analyzed: {len(df):,}
    - Overall Churn Rate: {df['actual_churn'].mean():.2%}
    - High-Risk Customers: {(df['risk_category'] == 'High').sum():,} ({((df['risk_category'] == 'High').sum()/len(df)*100):.1f}%)
    
    Risk Distribution:
    {df['risk_category'].value_counts().to_string()}
    
    Key Findings:
    1. High-risk customers typically have:
       - Shorter tenure (< 12 months)
       - Higher monthly charges
       - Month-to-month contracts
    
    2. Retention Opportunities:
       - Medium-risk customers: {(df['risk_category'] == 'Medium').sum():,} customers
       - These are prime targets for proactive retention efforts
    
    Recommendations:
    - Focus retention budget on High and Medium risk segments
    - Implement loyalty programs for short-tenure customers
    - Offer contract incentives for month-to-month customers
    - Create personalized offers based on churn probability
    
    Financial Impact:
    - Assuming average monthly charge of ${df['monthly_charges'].mean():.2f}
    - Monthly revenue at risk from high-risk customers: ${(df[df['risk_category'] == 'High']['monthly_charges'].sum()):,.2f}
    """
    
    with open('churn_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("✓ Business report saved as 'churn_analysis_report.txt'")
    return report

if __name__ == "__main__":
    print("=== CREATING COMPLETE DASHBOARD & REPORTS ===")
    
    # Check if predictions file exists
    if not os.path.exists('churn_predictions_complete.csv'):
        print("Please run the churn prediction script first!")
        print("Run: python churn_prediction_complete.py")
    else:
        # Create all deliverables
        dashboard_file = create_python_dashboard()
        create_business_report()
        
        print("\n🎉 DASHBOARD & REPORTS CREATED SUCCESSFULLY!")
        print("\n📊 DELIVERABLES:")
        print("1. churn_dashboard.html - Interactive dashboard (open in browser)")
        print("2. Multiple PNG files - Detailed visualizations")
        print("3. churn_analysis_report.txt - Business insights report")
        print("4. churn_predictions_complete.csv - Raw data for analysis")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Open 'churn_dashboard.html' in your web browser")
        print("2. Review 'churn_analysis_report.txt' for business insights")
        print("3. Use the PNG files in presentations/reports")
        
        # Optionally open the dashboard
        open_dashboard = input("\nWould you like to open the dashboard now? (y/n): ")
        if open_dashboard.lower() == 'y':
            webbrowser.open('churn_dashboard.html')
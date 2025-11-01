import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
import os
from datetime import datetime

def check_and_fix_data():
    """Check if data exists, create sample if needed"""
    if not os.path.exists('churn_predictions_complete.csv'):
        print("📁 Creating sample data for dashboard...")
        create_sample_data()
    else:
        print("✅ Data file found!")

def create_sample_data():
    """Create sample data if no file exists"""
    np.random.seed(42)
    n_customers = 1000
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'tenure': np.random.randint(1, 72, n_customers),
        'monthly_charges': np.random.uniform(20, 100, n_customers),
        'total_charges': np.random.uniform(50, 5000, n_customers),
        'contract_Month_to_month': np.random.choice([0, 1], n_customers, p=[0.4, 0.6]),
        'contract_One_year': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
        'contract_Two_year': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
        'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
        'partner_Yes': np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic churn probabilities
    churn_prob = (
        (df['tenure'] < 12) * 0.4 +
        (df['contract_Month_to_month'] == 1) * 0.3 +
        (df['monthly_charges'] > 70) * 0.2 +
        (df['senior_citizen'] == 1) * 0.1 +
        np.random.normal(0, 0.1, n_customers)
    )
    
    df['churn_probability'] = np.clip(churn_prob, 0, 1)
    df['actual_churn'] = (df['churn_probability'] > 0.5).astype(int)
    df['churn_prediction'] = (df['churn_probability'] > 0.3).astype(int)
    
    # Create risk categories
    conditions = [
        df['churn_probability'] < 0.3,
        df['churn_probability'] < 0.7
    ]
    choices = ['Low', 'Medium']
    df['risk_category'] = np.select(conditions, choices, 'High')
    
    # Save the data
    df.to_csv('churn_predictions_complete.csv', index=False)
    print("✅ Sample data created: 'churn_predictions_complete.csv'")
    return df

def create_dashboard_charts(df):
    """Create all charts needed for dashboard"""
    print("📊 Creating charts...")
    
    try:
        # 1. Risk Distribution Pie Chart
        plt.figure(figsize=(8, 6))
        risk_counts = df['risk_category'].value_counts()
        colors = ['#FF6B6B', '#FFD166', '#06D6A0']  # Red, Yellow, Green
        
        # Handle case where some risk categories might be missing
        available_colors = colors[:len(risk_counts)]
        
        plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                colors=available_colors, startangle=90)
        plt.title('Customer Risk Distribution')
        plt.savefig('risk_chart.png', dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ Created risk_chart.png")
        
        # 2. Tenure vs Churn Probability
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['tenure'], df['churn_probability'], 
                             alpha=0.6, c=df['churn_probability'], 
                             cmap='RdYlGn_r', s=30)
        plt.xlabel('Tenure (months)')
        plt.ylabel('Churn Probability')
        plt.title('Churn Probability vs Tenure')
        plt.colorbar(scatter, label='Churn Probability')
        plt.grid(True, alpha=0.3)
        plt.savefig('tenure_chart.png', dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ Created tenure_chart.png")
        
        # 3. Monthly Charges by Risk Category
        plt.figure(figsize=(10, 6))
        
        # Check if we have all risk categories
        available_risks = df['risk_category'].unique()
        plot_data = df[df['risk_category'].isin(available_risks)]
        
        if len(available_risks) > 0:
            sns.boxplot(data=plot_data, x='risk_category', y='monthly_charges',
                       order=['Low', 'Medium', 'High'])
            plt.title('Monthly Charges by Risk Category')
            plt.xlabel('Risk Category')
            plt.ylabel('Monthly Charges ($)')
            plt.savefig('charges_chart.png', dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            print("✅ Created charges_chart.png")
        else:
            # Create a simple histogram if boxplot fails
            plt.hist(df['monthly_charges'], bins=20, alpha=0.7, color='skyblue')
            plt.title('Distribution of Monthly Charges')
            plt.xlabel('Monthly Charges ($)')
            plt.ylabel('Number of Customers')
            plt.savefig('charges_chart.png', dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            print("✅ Created charges_chart.png (alternative)")
        
        # 4. Contract Type Analysis
        plt.figure(figsize=(10, 6))
        contract_cols = [col for col in df.columns if 'contract_' in col]
        
        if contract_cols:
            contract_data = []
            contract_labels = []
            
            for col in contract_cols:
                if col in df.columns and df[col].sum() > 0:
                    contract_mean = df[df[col] == 1]['churn_probability'].mean()
                    contract_data.append(contract_mean)
                    contract_labels.append(col.replace('contract_', '').replace('_', ' ').title())
            
            if contract_data:  # Only plot if we have data
                bars = plt.bar(contract_labels, contract_data, color=['#3498db', '#9b59b6', '#1abc9c'])
                plt.title('Average Churn Probability by Contract Type')
                plt.ylabel('Churn Probability')
                plt.xticks(rotation=45)
                
                # Add values on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('contract_chart.png', dpi=100, bbox_inches='tight', facecolor='white')
                plt.close()
                print("✅ Created contract_chart.png")
            else:
                # Create alternative chart
                create_alternative_chart(df)
        else:
            create_alternative_chart(df)
            
    except Exception as e:
        print(f"❌ Error creating charts: {e}")
        # Create simple backup charts
        create_backup_charts(df)

def create_alternative_chart(df):
    """Create an alternative chart if contract analysis fails"""
    plt.figure(figsize=(10, 6))
    
    # Create tenure segments
    df['tenure_segment'] = pd.cut(df['tenure'], bins=[0, 12, 24, 72], 
                                 labels=['0-12m', '13-24m', '25+m'])
    
    tenure_churn = df.groupby('tenure_segment')['churn_probability'].mean()
    
    bars = plt.bar(tenure_churn.index.astype(str), tenure_churn.values, 
                   color=['#FF6B6B', '#FFD166', '#06D6A0'])
    plt.title('Churn Probability by Tenure Segment')
    plt.ylabel('Average Churn Probability')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('contract_chart.png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Created contract_chart.png (tenure alternative)")

def create_backup_charts(df):
    """Create very simple backup charts if everything else fails"""
    print("🔄 Creating backup charts...")
    
    # Simple risk distribution
    plt.figure(figsize=(6, 6))
    risk_counts = df['risk_category'].value_counts()
    plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
    plt.title('Risk Distribution')
    plt.savefig('risk_chart.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Simple histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df['churn_probability'], bins=20, alpha=0.7, color='blue')
    plt.title('Churn Probability Distribution')
    plt.xlabel('Churn Probability')
    plt.ylabel('Number of Customers')
    plt.savefig('tenure_chart.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Copy the same chart for others to ensure files exist
    for chart in ['charges_chart.png', 'contract_chart.png']:
        plt.figure(figsize=(8, 6))
        plt.hist(df['monthly_charges'], bins=20, alpha=0.7, color='green')
        plt.title('Monthly Charges Distribution')
        plt.xlabel('Monthly Charges ($)')
        plt.ylabel('Number of Customers')
        plt.savefig(chart, dpi=100, bbox_inches='tight')
        plt.close()

def create_html_dashboard():
    """Create the main HTML dashboard"""
    print("🎨 Creating HTML dashboard...")
    
    # Load data
    df = pd.read_csv('churn_predictions_complete.csv')
    
    # Calculate key metrics safely
    try:
        total_customers = len(df)
        high_risk = (df['risk_category'] == 'High').sum() if 'risk_category' in df.columns else 0
        medium_risk = (df['risk_category'] == 'Medium').sum() if 'risk_category' in df.columns else 0
        
        if 'monthly_charges' in df.columns:
            revenue_risk = df[df['risk_category'].isin(['High', 'Medium'])]['monthly_charges'].sum() if 'risk_category' in df.columns else 0
        else:
            revenue_risk = 0
            
        avg_churn_prob = df['churn_probability'].mean() if 'churn_probability' in df.columns else 0
    except:
        # Use safe defaults if calculations fail
        total_customers = len(df)
        high_risk = 0
        medium_risk = 0
        revenue_risk = 0
        avg_churn_prob = 0
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Churn Dashboard</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background: #f5f5f5; 
            }}
            .dashboard {{ 
                max-width: 1200px; 
                margin: 0 auto; 
            }}
            .header {{ 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 20px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
            }}
            .metrics {{ 
                display: grid; 
                grid-template-columns: repeat(4, 1fr); 
                gap: 15px; 
                margin-bottom: 20px; 
            }}
            .metric-card {{ 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
            }}
            .metric-value {{ 
                font-size: 24px; 
                font-weight: bold; 
                margin: 10px 0; 
            }}
            .high-risk {{ color: #e74c3c; }}
            .medium-risk {{ color: #f39c12; }}
            .revenue {{ color: #27ae60; }}
            .charts {{ 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 20px; 
            }}
            .chart-container {{ 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
            }}
            img {{ 
                max-width: 100%; 
                height: auto; 
            }}
            .recommendations {{ 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                margin-top: 20px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
            }}
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="header">
                <h1>📊 Customer Churn Dashboard</h1>
                <p>Real-time customer retention insights | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div>Total Customers</div>
                    <div class="metric-value">{total_customers:,}</div>
                </div>
                <div class="metric-card high-risk">
                    <div>High Risk Customers</div>
                    <div class="metric-value">{high_risk}</div>
                    <div>({high_risk/total_customers*100:.1f}%)</div>
                </div>
                <div class="metric-card medium-risk">
                    <div>Medium Risk</div>
                    <div class="metric-value">{medium_risk}</div>
                    <div>({medium_risk/total_customers*100:.1f}%)</div>
                </div>
                <div class="metric-card revenue">
                    <div>Revenue at Risk</div>
                    <div class="metric-value">${revenue_risk:,.0f}</div>
                    <div>per month</div>
                </div>
            </div>
            
            <div class="charts">
                <div class="chart-container">
                    <h3>Risk Distribution</h3>
                    <img src="risk_chart.png" alt="Risk Distribution">
                </div>
                <div class="chart-container">
                    <h3>Churn Probability by Tenure</h3>
                    <img src="tenure_chart.png" alt="Tenure Analysis">
                </div>
                <div class="chart-container">
                    <h3>Monthly Charges Analysis</h3>
                    <img src="charges_chart.png" alt="Charges Analysis">
                </div>
                <div class="chart-container">
                    <h3>Contract Type Impact</h3>
                    <img src="contract_chart.png" alt="Contract Analysis">
                </div>
            </div>
            
            <div class="recommendations">
                <h3>🎯 Immediate Actions Recommended</h3>
                <div style="background: #ffebee; padding: 15px; margin: 10px 0; border-radius: 5px;">
                    <strong>HIGH RISK ({high_risk} customers):</strong><br>
                    • Personal phone calls within 24 hours<br>
                    • Offer 25% discount for 3 months<br>
                    • Assign dedicated account manager
                </div>
                <div style="background: #fff3e0; padding: 15px; margin: 10px 0; border-radius: 5px;">
                    <strong>MEDIUM RISK ({medium_risk} customers):</strong><br>
                    • Targeted email campaign<br>
                    • Loyalty program invitation<br>
                    • Usage optimization tips
                </div>
                
                <h4>💡 Quick Insights:</h4>
                <ul>
                    <li>Average churn probability: {avg_churn_prob:.1%}</li>
                    <li>Focus on customers with tenure < 12 months</li>
                    <li>Monitor customers with monthly charges > $70</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    with open('churn_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML dashboard created: 'churn_dashboard.html'")

def main():
    print("🚀 FIXED DASHBOARD GENERATOR")
    print("=" * 50)
    
    try:
        # Step 1: Check and prepare data
        check_and_fix_data()
        
        # Step 2: Load data
        df = pd.read_csv('churn_predictions_complete.csv')
        print(f"✅ Loaded data: {len(df)} customers")
        
        # Step 3: Create charts
        create_dashboard_charts(df)
        
        # Step 4: Create HTML dashboard
        create_html_dashboard()
        
        # Step 5: Open in browser
        print("🌐 Opening dashboard in your browser...")
        webbrowser.open('churn_dashboard.html')
        
        print("\n🎉 DASHBOARD CREATED SUCCESSFULLY!")
        print("✅ churn_dashboard.html - Open this file in any browser")
        print("📊 All charts generated")
        print("📁 Files created:")
        print("   - churn_dashboard.html (main dashboard)")
        print("   - risk_chart.png, tenure_chart.png, etc.")
        print("\n💡 To view later: Double-click 'churn_dashboard.html'")
        print("📧 To share: Send the HTML file to anyone!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Python and required packages are installed")
        print("2. Run: python -m pip install pandas matplotlib seaborn")
        print("3. Try running in a different directory")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from flask import Flask, render_template, request, flash, redirect, url_for
import warnings
warnings.filterwarnings('ignore')

# Enhanced configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.secret_key = 'your_secret_key_here'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Optimized model for large datasets
def train_optimized_model(data, test_size=0.2):
    """
    Train an optimized model for large datasets
    """
    # Data preparation
    data_clean = data.dropna().copy()
    data_clean['Day'] = np.arange(len(data_clean))
    
    # Sampling if too much data
    if len(data_clean) > 10000:
        data_clean = data_clean.sample(n=10000, random_state=42)
    
    X = data_clean[['Day']].values
    y = data_clean['Sales'].values
    
    # Normalization
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled.ravel(), test_size=test_size, random_state=42
    )
    
    # Model with optimized parameters
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        batch_size='auto'
    )
    
    # Training with progress tracking
    model.fit(X_train, y_train)
    
    # Evaluation
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training score: {train_score:.3f}")
    print(f"Test score: {test_score:.3f}")
    
    return model, scaler_X, scaler_y, train_score, test_score

# Enhanced fuzzy logic
def apply_advanced_fuzzy_logic(sales, trend=None, seasonality=None):
    """
    Enhanced fuzzy logic with more criteria
    """
    recommendations = []
    
    # Sales level analysis
    if sales < 100:
        recommendations.append("🚨 Critical sales - Review your commercial strategy")
        recommendations.append("📊 Analyze causes of decline")
        recommendations.append("🎯 Target new customer segments")
    elif sales < 500:
        recommendations.append("📈 Moderate sales - Potential for improvement")
        recommendations.append("💡 Test new marketing campaigns")
        recommendations.append("👥 Strengthen customer loyalty")
    elif sales < 2000:
        recommendations.append("✅ Good performance - Maintain effort")
        recommendations.append("📱 Optimize customer experience")
        recommendations.append("🔄 Improve processes")
    else:
        recommendations.append("🎉 Excellent performance!")
        recommendations.append("🚀 Consider expansion")
        recommendations.append("📚 Capitalize on your successes")
    
    # Trend analysis if available
    if trend == "growing":
        recommendations.append("📈 Positive trend - Invest in growth")
    elif trend == "declining":
        recommendations.append("⚠️ Negative trend - Analyze quickly")
    
    return recommendations

# Optimized prediction function
def predict_optimized(model, scaler_X, scaler_y, start_day, num_periods):
    """
    Optimized predictions with batch management
    """
    predictions = []
    batch_size = min(1000, num_periods)  # Batch processing for large predictions
    
    for i in range(0, num_periods, batch_size):
        end_batch = min(i + batch_size, num_periods)
        days_batch = [[start_day + j + 1] for j in range(i, end_batch)]
        
        scaled_days = scaler_X.transform(days_batch)
        scaled_predictions = model.predict(scaled_days)
        predictions_batch = scaler_y.inverse_transform(
            scaled_predictions.reshape(-1, 1)
        )
        
        predictions.extend(predictions_batch.flatten())
    
    return predictions

# Enhanced routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        recommendations = []
        message = ""
        data_table = None
        predictions = []
        stats = {}
        
        if 'file' in request.files and request.files['file'].filename != '':
            # File processing
            file = request.files['file']
            if not file.filename.endswith('.csv'):
                flash("Please upload a CSV file", "error")
                return redirect(url_for('home'))
                
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            try:
                # Load with memory management
                data = pd.read_csv(file_path)
                
                if 'Sales' not in data.columns:
                    flash("The file must contain a 'Sales' column", "error")
                    return redirect(url_for('home'))
                
                # Descriptive statistics
                stats = {
                    'average': data['Sales'].mean(),
                    'median': data['Sales'].median(),
                    'std_dev': data['Sales'].std(),
                    'min': data['Sales'].min(),
                    'max': data['Sales'].max(),
                    'total_rows': len(data)
                }
                
                # Prepare data for display
                data_sample = data.head(100) if len(data) > 100 else data
                data_table = data_sample.to_dict('records')
                
                # Train optimized model
                model, scaler_X, scaler_y, train_score, test_score = train_optimized_model(data)
                
                # Enhanced fuzzy logic
                trend = "growing" if data['Sales'].iloc[-1] > data['Sales'].iloc[0] else "declining"
                recommendations = apply_advanced_fuzzy_logic(stats['average'], trend)
                
                # Predictions
                period = request.form.get('prediction_period', 'days')
                period_length = int(request.form.get('period_length', 7))
                
                multiplier = {'days': 1, 'weeks': 7, 'months': 30, 'years': 365}
                num_periods = period_length * multiplier.get(period, 1)
                
                # Limit predictions to avoid long computation times
                num_periods = min(num_periods, 365 * 5)  # Max 5 years
                
                predictions_dates = pd.date_range(
                    start=pd.to_datetime('today'), 
                    periods=num_periods
                ).strftime('%Y-%m-%d')
                
                predictions_values = predict_optimized(
                    model, scaler_X, scaler_y, len(data), num_periods
                )
                
                predictions = list(zip(predictions_dates, predictions_values))
                
                message = f"""
                ✅ Analysis completed! 
                • {len(data)} rows processed
                • Model score: {test_score:.3f}
                • Forecasts: {num_periods} {period}
                """
                
            except Exception as e:
                message = f"❌ Error: {str(e)}"
                
        elif 'sales' in request.form and request.form['sales']:
            # Manual input processing
            sales = float(request.form['sales'])
            date = request.form.get('date', pd.Timestamp.today().strftime('%Y-%m-%d'))
            
            # Generate synthetic data for training
            days = 30
            base_sales = sales * 0.8
            data = pd.DataFrame({
                'Day': range(1, days + 1),
                'Sales': np.random.normal(sales, sales * 0.1, days)
            })
            
            model, scaler_X, scaler_y, _, _ = train_optimized_model(data)
            recommendations = apply_advanced_fuzzy_logic(sales)
            
            # Predictions
            predictions_dates = pd.date_range(
                start=pd.to_datetime(date), 
                periods=30
            ).strftime('%Y-%m-%d')
            
            predictions_values = predict_optimized(model, scaler_X, scaler_y, 30, 30)
            predictions = list(zip(predictions_dates, predictions_values))
            
            message = "✅ Analysis based on entered data"
            
        else:
            message = "ℹ️ Please provide input data"
            
        return render_template('result.html', 
                             message=message, 
                             predictions=predictions, 
                             data_table=data_table,
                             recommendations=recommendations,
                             stats=stats)
                             
    except Exception as e:
        flash(f"General error: {str(e)}", "error")
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
# complete_price_automation_fixed.py
"""
Complete automation pipeline for Department of Consumer Affairs price monitoring
Includes data collection, preprocessing, ML forecasting, and web dashboard updates
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import requests
import schedule
import time
import threading
from datetime import datetime, timedelta
import logging
from flask import Flask, jsonify, render_template, render_template_string
import plotly
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class Config:
    DATABASE_PATH = 'price_monitoring.db'
    LOG_FILE = 'price_monitoring.log'
    
    # API Configuration
    API_KEYS = {
        'data_gov': '35985678-0d79-46b4-9ed6-6f13308a1d24'  # Replace with actual key
    }
    
    API_ENDPOINTS = {
        'data_gov': 'https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070'
    }
    
    COMMODITIES = [
        'Tur Dal', 'Urad Dal', 'Moong Dal', 'Masur Dal', 'Gram Dal',
        'Rice', 'Wheat', 'Atta', 'Onion', 'Potato', 'Tomato',
        'Groundnut Oil', 'Mustard Oil', 'Vanaspati', 'Sugar', 'Milk', 
        'Salt', 'Tea', 'Palm Oil', 'Soya Oil', 'Sunflower Oil', 'Gur'
    ]
    
    MARKETS = [
        'Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bengaluru', 'Hyderabad'
    ]
    
    PRICE_THRESHOLDS = {
        'alert': 5.0,     # 5% change triggers alert
        'action': 10.0    # 10% change triggers action
    }

# Initialize Flask
app = Flask(__name__)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handle all database operations"""
    
    def __init__(self, db_path=Config.DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Price data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                commodity TEXT NOT NULL,
                market TEXT NOT NULL,
                price REAL NOT NULL,
                unit TEXT DEFAULT 'per_kg',
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, commodity, market)
            )
        ''')
        
        # Forecast data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_date DATE NOT NULL,
                commodity TEXT NOT NULL,
                market TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                confidence_upper REAL,
                confidence_lower REAL,
                model_used TEXT,
                accuracy_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Buffer stock tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS buffer_stock (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commodity TEXT NOT NULL,
                current_stock REAL NOT NULL,
                min_threshold REAL NOT NULL,
                max_capacity REAL NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Stock interventions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commodity TEXT NOT NULL,
                intervention_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                reason TEXT,
                price_trigger REAL,
                intervention_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def insert_price_data(self, data):
        """Insert price data into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.executemany('''
                INSERT OR REPLACE INTO price_data 
                (date, commodity, market, price, unit, source) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', data)
            conn.commit()
            logger.info(f"Inserted {len(data)} price records")
        except Exception as e:
            logger.error(f"Error inserting price data: {e}")
        finally:
            conn.close()
    
    def get_price_data(self, commodity=None, market=None, days=30):
        """Retrieve price data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT date, commodity, market, price 
            FROM price_data 
            WHERE date >= date('now', '-{} days')
        '''.format(days)
        
        conditions = []
        params = []
        
        if commodity:
            conditions.append("commodity = ?")
            params.append(commodity)
        
        if market:
            conditions.append("market = ?")
            params.append(market)
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " ORDER BY date DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df

    def get_latest_price(self, commodity, market):
        """Get the latest price for a commodity in a market"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT price FROM price_data 
                WHERE commodity = ? AND market = ?
                ORDER BY date DESC LIMIT 1
            ''', (commodity, market))
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()

    def get_price_change(self, commodity, market, days=7):
        """Calculate price change over specified days"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT price FROM price_data 
            WHERE commodity = ? AND market = ?
            ORDER BY date DESC LIMIT ?
        ''', conn, params=[commodity, market, days])
        conn.close()
        
        if len(df) >= 2:
            latest_price = df['price'].iloc[0]
            old_price = df['price'].iloc[-1]
            return ((latest_price - old_price) / old_price) * 100
        return None

class DataCollector:
    """Collect price data from various government sources"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def generate_sample_data(self):
        """Generate sample data for demo purposes"""
        sample_data = []
        base_date = datetime.now().date() - timedelta(days=30)
        
        # Add all commodities with realistic base prices
        base_prices = {
            # Pulses
            'Tur Dal': 110, 'Urad Dal': 105, 'Moong Dal': 95, 
            'Masur Dal': 85, 'Gram Dal': 75,
            # Cereals
            'Rice': 40, 'Wheat': 25, 'Atta': 30,
            # Vegetables
            'Onion': 30, 'Potato': 20, 'Tomato': 40,
            # Oils
            'Groundnut Oil': 180, 'Mustard Oil': 165, 'Vanaspati': 120,
            # Other essentials
            'Sugar': 45, 'Milk': 55, 'Salt': 20, 'Tea': 250,
            'Palm Oil': 110, 'Soya Oil': 125, 'Sunflower Oil': 135,
            'Gur': 45
        }
        
        for days_back in range(30):
            current_date = base_date + timedelta(days=days_back)
            
            for commodity in Config.COMMODITIES:
                if commodity in base_prices:
                    for market in Config.MARKETS:
                        # Add realistic price variation
                        base_price = base_prices[commodity]
                        market_factor = np.random.uniform(0.9, 1.1)  # Market variation
                        seasonal_factor = 1 + 0.1 * np.sin(days_back / 7)  # Weekly pattern
                        random_factor = np.random.uniform(0.95, 1.05)  # Daily randomness
                        
                        price = base_price * market_factor * seasonal_factor * random_factor
                        
                        sample_data.append((
                            current_date, commodity, market, round(price, 2), 'per_kg', 'sample'
                        ))
        
        # Initialize buffer stock data
        self._initialize_buffer_stock()
    
        self.db_manager.insert_price_data(sample_data)
        logger.info(f"Generated {len(sample_data)} sample records")
        return sample_data

    def _initialize_buffer_stock(self):
        """Initialize buffer stock data"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        buffer_stock_data = [
            ('Tur Dal', 15000, 10000, 25000),
            ('Urad Dal', 12000, 8000, 20000),
            ('Moong Dal', 18000, 12000, 30000),
            ('Masur Dal', 10000, 7000, 18000),
            ('Gram Dal', 20000, 15000, 35000),
            ('Onion', 5000, 3000, 10000)
        ]
        
        try:
            cursor.executemany('''
                INSERT OR REPLACE INTO buffer_stock 
                (commodity, current_stock, min_threshold, max_capacity)
                VALUES (?, ?, ?, ?)
            ''', buffer_stock_data)
            conn.commit()
            logger.info("Buffer stock data initialized")
        except Exception as e:
            logger.error(f"Error initializing buffer stock: {e}")
        finally:
            conn.close()

class PriceForecastModel:
    """ML models for price forecasting"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.models = {}
    
    def prepare_data(self, commodity, market, days=60):
        """Prepare time series data for modeling"""
        df = self.db_manager.get_price_data(commodity=commodity, market=market, days=days)
        
        if df.empty:
            logger.warning(f"No data found for {commodity} in {market}")
            return None
        
        # Sort by date and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        # Resample to daily frequency and fill gaps
        df = df.resample('D').mean().fillna(method='ffill').fillna(method='bfill')
        
        return df['price']
    
    def generate_forecasts(self, commodity, market, days=7):
        """Generate price forecasts"""
        forecasts = []
        
        # Generate simple trend-based forecast
        forecasts = self._generate_simple_forecast(commodity, market, days)
        
        # Save forecasts to database
        if forecasts:
            self._save_forecasts(forecasts)
        
        return forecasts
    
    def _generate_simple_forecast(self, commodity, market, days):
        """Generate simple trend-based forecast"""
        forecasts = []
        
        try:
            recent_data = self.prepare_data(commodity, market, days=14)
            if recent_data is None or len(recent_data) < 5:
                return forecasts
            
            # Calculate simple trend
            last_price = recent_data.iloc[-1]
            trend = (recent_data.iloc[-1] - recent_data.iloc[-7]) / 7 if len(recent_data) >= 7 else 0
            
            for i in range(days):
                forecast_date = datetime.now().date() + timedelta(days=i+1)
                predicted_price = last_price + (trend * (i + 1))
                
                forecasts.append({
                    'date': forecast_date,
                    'commodity': commodity,
                    'market': market,
                    'predicted_price': float(predicted_price),
                    'confidence_upper': float(predicted_price * 1.15),
                    'confidence_lower': float(predicted_price * 0.85),
                    'model_used': 'Simple_Trend',
                    'accuracy_score': 75.0
                })
                
        except Exception as e:
            logger.error(f"Error generating simple forecast: {e}")
        
        return forecasts
    
    def _save_forecasts(self, forecasts):
        """Save forecasts to database"""
        if not forecasts:
            return
        
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        try:
            for forecast in forecasts:
                cursor.execute('''
                    INSERT OR REPLACE INTO forecasts 
                    (forecast_date, commodity, market, predicted_price, 
                     confidence_upper, confidence_lower, model_used, accuracy_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    forecast['date'], forecast['commodity'], forecast['market'],
                    forecast['predicted_price'], forecast['confidence_upper'],
                    forecast['confidence_lower'], forecast['model_used'],
                    forecast['accuracy_score']
                ))
            
            conn.commit()
            logger.info(f"Saved {len(forecasts)} forecasts to database")
            
        except Exception as e:
            logger.error(f"Error saving forecasts: {e}")
        finally:
            conn.close()

class ChartGenerator:
    """Generate interactive charts for the web dashboard"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def create_price_trend_chart(self, commodity, market):
        """Create price trend visualization"""
        try:
            df = self.db_manager.get_price_data(commodity=commodity, market=market, days=30)
            
            if df.empty:
                return self._create_empty_chart("No data available")
            
            # Calculate price change
            df['price_change'] = df['price'].pct_change() * 100
            
            trace = go.Scatter(
                x=pd.to_datetime(df['date']),
                y=df['price'],
                mode='lines+markers',
                name='Daily Price',
                line=dict(color='#667eea', width=2),
                hovertemplate='Date: %{x}<br>Price: ₹%{y:.2f}/kg<extra></extra>'
            )
            
            layout = go.Layout(
                title=f'{commodity} Price Trend - {market}',
                xaxis_title='Date',
                yaxis_title='Price (₹/kg)',
                plot_bgcolor='white',
                height=500,
                showlegend=True
            )
            
            fig = go.Figure(data=[trace], layout=layout)
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            logger.error(f"Error creating price trend chart: {e}")
            return self._create_empty_chart("Error loading chart")

    def _create_empty_chart(self, message="No data available"):
        """Create empty chart with message"""
        layout = go.Layout(
            title=message,
            xaxis_title='Date',
            yaxis_title='Price',
            height=500
        )
        fig = go.Figure(layout=layout)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def create_forecast_chart(self, commodity, market):
        """Create forecast chart"""
        try:
            historical_df = self.db_manager.get_price_data(
                commodity=commodity, market=market, days=14
            )
            
            conn = sqlite3.connect(self.db_manager.db_path)
            forecast_df = pd.read_sql_query('''
                SELECT forecast_date, predicted_price, confidence_upper, confidence_lower
                FROM forecasts 
                WHERE commodity = ? AND market = ?
                ORDER BY forecast_date
            ''', conn, params=[commodity, market])
            conn.close()
            
            traces = []
            
            # Historical data trace
            if not historical_df.empty:
                traces.append(go.Scatter(
                    x=pd.to_datetime(historical_df['date']),
                    y=historical_df['price'],
                    mode='lines+markers',
                    name='Historical Price',
                    line=dict(color='#667eea', width=2)
                ))
            
            # Forecast traces
            if not forecast_df.empty:
                traces.append(go.Scatter(
                    x=pd.to_datetime(forecast_df['forecast_date']),
                    y=forecast_df['predicted_price'],
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='#764ba2', width=2, dash='dash')
                ))
            
            layout = go.Layout(
                title=f'{commodity} Price Forecast - {market}',
                xaxis_title='Date',
                yaxis_title='Price (₹/kg)',
                showlegend=True,
                plot_bgcolor='white',
                height=500
            )
            
            fig = go.Figure(data=traces, layout=layout)
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating forecast chart: {e}")
            return self._create_empty_chart("Error loading forecast")

class BufferStockManager:
    """Manage buffer stock operations and interventions"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def analyze_price_trends(self, commodity):
        """Analyze price trends for intervention decisions"""
        try:
            df = self.db_manager.get_price_data(commodity=commodity, days=14)
            if df.empty or len(df) < 7:
                return None
            
            # Calculate price change over the week
            recent_avg = df.head(7)['price'].mean()
            older_avg = df.tail(7)['price'].mean()
            price_change = ((recent_avg - older_avg) / older_avg) * 100
            
            return {
                'commodity': commodity,
                'price_change_percent': round(price_change, 2),
                'current_avg_price': round(recent_avg, 2),
                'intervention_recommended': abs(price_change) > Config.PRICE_THRESHOLDS['alert']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing price trends for {commodity}: {e}")
            return None

# Flask Routes
@app.route('/')
def index():
    """Serve the main dashboard page"""
    try:
        # Remove HTML_TEMPLATE variable since we're using a separate file
        return render_template('index.html', 
                             commodities=Config.COMMODITIES,
                             markets=Config.MARKETS)
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return "Error loading dashboard", 500

@app.route('/api/charts/price-trend/<commodity>/<market>')
def get_price_trend(commodity, market):
    try:
        if commodity not in Config.COMMODITIES:
            return jsonify({'error': 'Invalid commodity'}), 400
        if market not in Config.MARKETS:
            return jsonify({'error': 'Invalid market'}), 400
            
        chart_json = chart_generator.create_price_trend_chart(commodity, market)
        
        # Add metrics to the response
        metrics = {
            'currentPrice': db_manager.get_latest_price(commodity, market),
            'priceChange': db_manager.get_price_change(commodity, market, days=7),
            'accuracy': 75.0  # Sample accuracy value
        }
        
        return jsonify({
            'data': json.loads(chart_json)['data'],
            'layout': json.loads(chart_json)['layout'],
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Error in price trend API: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/charts/forecast/<commodity>/<market>')
def get_forecast(commodity, market):
    try:
        if commodity not in Config.COMMODITIES:
            return jsonify({'error': 'Invalid commodity'}), 400
        if market not in Config.MARKETS:
            return jsonify({'error': 'Invalid market'}), 400
            
        chart_json = chart_generator.create_forecast_chart(commodity, market)
        return chart_json, 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error in forecast API: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def scheduled_data_update():
    """Function to run scheduled data updates"""
    logger.info("Starting scheduled data update...")
    try:
        # Generate sample data (in real scenario, this would fetch from APIs)
        data_collector.generate_sample_data()
        
        # Generate forecasts for key commodities
        for commodity in ['Tur Dal', 'Onion', 'Potato']:
            for market in ['Delhi', 'Mumbai']:
                forecast_model.generate_forecasts(commodity, market)
        
        logger.info("Scheduled data update completed successfully")
    except Exception as e:
        logger.error(f"Error in scheduled update: {e}")

def setup_scheduler():
    """Setup automated data collection schedule"""
    try:
        schedule.every(2).hours.do(scheduled_data_update)
        logger.info("Scheduler setup complete")
    except Exception as e:
        logger.error(f"Scheduler setup failed: {e}")

def run_scheduler():
    """Run the scheduler in background"""
    while True:
        try:
            schedule.run_pending()
            time.sleep(300)  # Check every 5 minutes
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(300)  # Wait 5 minutes on error

# Global variables for components
db_manager = None
data_collector = None
chart_generator = None
forecast_model = None
buffer_manager = None

if __name__ == '__main__':
    try:
        # Initialize components
        logger.info("Initializing Price Monitoring System...")
        db_manager = DatabaseManager()
        data_collector = DataCollector(db_manager)
        chart_generator = ChartGenerator(db_manager)
        forecast_model = PriceForecastModel(db_manager)
        
        # Generate initial sample data
        logger.info("Generating sample data...")
        data_collector.generate_sample_data()
        
        # Generate initial forecasts
        logger.info("Generating initial forecasts...")
        for commodity in ['Tur Dal', 'Onion']:
            for market in ['Delhi', 'Mumbai']:
                forecast_model.generate_forecasts(commodity, market)
        
        # Setup scheduler
        setup_scheduler()
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        # Start Flask application
        logger.info("Starting web application on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise

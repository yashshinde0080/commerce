# app.py

from flask import Flask, render_template, jsonify, request
import json
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# --- Dummy Data for Demonstration ---
# In a real application, these would come from a database or a more complex data source.
COMMODITIES = ['Wheat', 'Rice', 'Sugar', 'Pulses', 'Edible Oil']
MARKETS = ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata', 'Chennai', 'Hyderabad']

def generate_dummy_price_data(commodity, market, days=30):
    """Generates dummy historical price data."""
    prices = []
    dates = []
    base_price = {
        'Wheat': 2500, 'Rice': 4000, 'Sugar': 40, 'Pulses': 70, 'Edible Oil': 120
    }.get(commodity, 100)
    
    # Introduce some market-specific variance
    market_variance = {
        'Delhi': 1.02, 'Mumbai': 1.05, 'Bengaluru': 0.98, 'Kolkata': 1.03, 'Chennai': 0.97, 'Hyderabad': 1.01
    }.get(market, 1.0)

    current_date = datetime.now()
    for i in range(days):
        date = current_date - timedelta(days=days - 1 - i)
        dates.append(date.strftime('%Y-%m-%d'))
        # Simulate price fluctuations
        price = base_price * market_variance * (1 + (random.random() - 0.5) * 0.1) # +/- 5% fluctuation
        prices.append(round(price, 2))
    
    # Calculate simple metrics for dummy data
    current_price = prices[-1]
    # Price change over last 7 days (or less if data is shorter)
    price_7_days_ago_idx = max(0, len(prices) - 7)
    price_7_days_ago = prices[price_7_days_ago_idx]
    price_change = ((current_price - price_7_days_ago) / price_7_days_ago) * 100 if price_7_days_ago != 0 else 0

    # Dummy forecast accuracy
    forecast_accuracy = random.uniform(75.0, 95.0)

    return {
        "dates": dates,
        "prices": prices,
        "currentPrice": current_price,
        "priceChange": price_change,
        "accuracy": forecast_accuracy
    }

def generate_dummy_forecast_data(commodity, market, historical_days=15, forecast_days=7):
    """Generates dummy forecast data."""
    # Use the same logic for historical part
    historical_data = generate_dummy_price_data(commodity, market, days=historical_days)
    
    forecast_dates = []
    forecast_prices = []
    last_historical_price = historical_data["prices"][-1]
    last_historical_date = datetime.strptime(historical_data["dates"][-1], '%Y-%m-%d')

    for i in range(1, forecast_days + 1):
        date = last_historical_date + timedelta(days=i)
        forecast_dates.append(date.strftime('%Y-%m-%d'))
        # Simple linear projection for forecast
        forecast_price = last_historical_price + (random.random() - 0.5) * 5 # Small random fluctuation
        forecast_prices.append(round(forecast_price, 2))

    return {
        "historical_dates": historical_data["dates"],
        "historical_prices": historical_data["prices"],
        "forecast_dates": forecast_dates,
        "forecast_prices": forecast_prices
    }

# --- Routes ---

@app.route('/')
def index():
    """Renders the main dashboard HTML page."""
    # Flask looks for templates in a 'templates' folder by default
    return render_template('index.html', commodities=COMMODITIES, markets=MARKETS)

@app.route('/api/charts/price-trend/<string:commodity>/<string:market>')
def price_trend_api(commodity, market):
    """API endpoint for price trend data."""
    data = generate_dummy_price_data(commodity, market)
    
    response_data = {
        "data": [
            {
                "x": data["dates"],
                "y": data["prices"],
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Actual Price",
                "line": {"color": "#4299e1", "width": 2},
                "marker": {"size": 6, "color": "#4299e1", "line": {"width": 1, "color": "white"}}
            },
        ],
        "layout": {
            "title": {
                "text": f"Price Trend for {commodity} in {market}",
                "font": {"size": 18, "color": "#2d3748", "family": "Inter, sans-serif"}
            },
            "xaxis": {
                "title": "Date",
                "showgrid": True,
                "gridcolor": "#e2e8f0",
                "tickfont": {"family": "Inter, sans-serif"}
            },
            "yaxis": {
                "title": "Price (₹)",
                "showgrid": True,
                "gridcolor": "#e2e8f0",
                "tickfont": {"family": "Inter, sans-serif"}
            },
            "margin": {"l": 50, "r": 50, "b": 50, "t": 50},
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "hovermode": "x unified",
            "font": {"family": "Inter, sans-serif"}
        },
        "metrics": {
            "currentPrice": data["currentPrice"],
            "priceChange": data["priceChange"],
            "accuracy": data["accuracy"]
        }
    }
    return jsonify(response_data)

@app.route('/api/charts/forecast/<string:commodity>/<string:market>')
def forecast_api(commodity, market):
    """API endpoint for forecast data."""
    data = generate_dummy_forecast_data(commodity, market)
    
    response_data = {
        "data": [
            {
                "x": data["historical_dates"],
                "y": data["historical_prices"],
                "type": "scatter",
                "mode": "lines",
                "name": "Historical Price",
                "line": {"color": "#4299e1", "width": 2},
                "hovertemplate": "<b>Date</b>: %{x}<br><b>Price</b>: ₹%{y:.2f}<extra></extra>"
            },
            {
                "x": data["forecast_dates"],
                "y": data["forecast_prices"],
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Forecast Price",
                "line": {"color": "#F59E0B", "dash": "dash", "width": 2},
                "marker": {"size": 6, "color": "#F59E0B", "line": {"width": 1, "color": "white"}},
                "hovertemplate": "<b>Date</b>: %{x}<br><b>Forecast</b>: ₹%{y:.2f}<extra></extra>"
            }
        ],
        "layout": {
            "title": {
                "text": f"Price Forecast for {commodity} in {market}",
                "font": {"size": 18, "color": "#2d3748", "family": "Inter, sans-serif"}
            },
            "xaxis": {
                "title": "Date",
                "showgrid": True,
                "gridcolor": "#e2e8f0",
                "tickfont": {"family": "Inter, sans-serif"}
            },
            "yaxis": {
                "title": "Price (₹)",
                "showgrid": True,
                "gridcolor": "#e2e8f0",
                "tickfont": {"family": "Inter, sans-serif"}
            },
            "margin": {"l": 50, "r": 50, "b": 50, "t": 50},
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "hovermode": "x unified",
            "font": {"family": "Inter, sans-serif"},
            "legend": {"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.7)", "bordercolor": "#e2e8f0", "borderwidth": 1}
        }
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True) # debug=True enables auto-reloading and better error messages

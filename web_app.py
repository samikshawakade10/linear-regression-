"""
House Price Prediction - Modern Web Dashboard
----------------------------------------------
Runs a self-contained local web server (built-in Python http.server)
with an interactive HTML5/Tailwind dashboard and REST API endpoints.
Zero external web framework dependencies needed.
"""

import os
import sys
import json
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "house_price_regression_dataset.csv")

from predict import load_trained_model, predict_price

PIPELINE = None
METADATA = {}


def load_globals():
    global PIPELINE, METADATA
    try:
        if os.path.isfile(MODEL_PATH):
            PIPELINE = load_trained_model(MODEL_PATH)
        if os.path.isfile(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                METADATA = json.load(f)
    except Exception as e:
        print(f"Error loading model globals: {e}")


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>House Price Prediction — Linear Regression Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    .gradient-header {
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen antialiased flex flex-col">

  <!-- Header -->
  <header class="gradient-header text-white shadow-lg border-b border-slate-700">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      <div>
        <div class="flex items-center gap-3">
          <div class="p-2.5 bg-blue-600 rounded-lg text-white shadow-md">
            <i class="fa-solid fa-house-chimney text-xl"></i>
          </div>
          <div>
            <h1 class="text-2xl font-bold tracking-tight">House Price Prediction System</h1>
            <p class="text-sm text-slate-400">Ordinary Least Squares (OLS) Linear Regression Machine Learning Model</p>
          </div>
        </div>
      </div>
      <div class="flex flex-wrap items-center gap-2">
        <span class="px-3 py-1.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-mono text-xs font-semibold">
          <i class="fa-solid fa-check-circle mr-1"></i> R² Score: 99.84%
        </span>
        <span class="px-3 py-1.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 font-mono text-xs font-semibold">
          MAE: $8,174.58
        </span>
        <span class="px-3 py-1.5 rounded-full bg-purple-500/20 text-purple-400 border border-purple-500/30 font-mono text-xs font-semibold">
          RMSE: $10,071.48
        </span>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full space-y-8">
    
    <!-- Top Grid: Interactive Predictor & Price Valuation Card -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      <!-- Input Form (7 cols) -->
      <div class="lg:col-span-7 bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
        <div class="flex items-center justify-between pb-4 mb-6 border-b border-slate-100">
          <div>
            <h2 class="text-lg font-bold text-slate-900">Property Specifications</h2>
            <p class="text-xs text-slate-500">Configure parameters to estimate real estate valuation</p>
          </div>
          <button onclick="resetDefaults()" class="text-xs text-blue-600 hover:text-blue-800 font-medium flex items-center gap-1">
            <i class="fa-solid fa-rotate-left"></i> Reset Defaults
          </button>
        </div>

        <form id="prediction-form" class="space-y-5" onsubmit="event.preventDefault(); submitPrediction();">
          
          <!-- Square Footage Slider + Input -->
          <div>
            <div class="flex justify-between text-xs font-semibold text-slate-700 mb-1">
              <label for="sqft"><i class="fa-solid fa-ruler-combined text-blue-500 mr-1"></i> Square Footage (sq ft)</label>
              <span id="sqft-val" class="font-mono text-blue-600 font-bold">2,800 sq ft</span>
            </div>
            <input type="range" id="sqft" min="500" max="10000" step="25" value="2800" 
              class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              oninput="syncVal('sqft', 'sqft-val', ' sq ft', true); submitPrediction();">
          </div>

          <!-- 2 Columns Grid for Bedrooms & Bathrooms -->
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-xs font-semibold text-slate-700 mb-1">
                <i class="fa-solid fa-bed text-blue-500 mr-1"></i> Bedrooms
              </label>
              <select id="bed" class="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none" onchange="submitPrediction();">
                <option value="1">1 Bedroom</option>
                <option value="2">2 Bedrooms</option>
                <option value="3" selected>3 Bedrooms</option>
                <option value="4">4 Bedrooms</option>
                <option value="5">5 Bedrooms</option>
                <option value="6">6 Bedrooms</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-semibold text-slate-700 mb-1">
                <i class="fa-solid fa-bath text-blue-500 mr-1"></i> Bathrooms
              </label>
              <select id="bath" class="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none" onchange="submitPrediction();">
                <option value="1">1 Bathroom</option>
                <option value="2" selected>2 Bathrooms</option>
                <option value="3">3 Bathrooms</option>
                <option value="4">4 Bathrooms</option>
              </select>
            </div>
          </div>

          <!-- Year Built & Lot Size -->
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-xs font-semibold text-slate-700 mb-1">
                <i class="fa-solid fa-calendar text-blue-500 mr-1"></i> Year Built
              </label>
              <input type="number" id="year" min="1950" max="2026" value="2000"
                class="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:outline-none"
                oninput="submitPrediction();">
            </div>
            <div>
              <label class="block text-xs font-semibold text-slate-700 mb-1">
                <i class="fa-solid fa-tree text-blue-500 mr-1"></i> Lot Size (Acres)
              </label>
              <input type="number" id="lot" step="0.1" min="0.1" max="20" value="2.5"
                class="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:outline-none"
                oninput="submitPrediction();">
            </div>
          </div>

          <!-- Garage Size & Neighborhood Quality -->
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-xs font-semibold text-slate-700 mb-1">
                <i class="fa-solid fa-warehouse text-blue-500 mr-1"></i> Garage Size (Cars)
              </label>
              <select id="garage" class="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none" onchange="submitPrediction();">
                <option value="0">No Garage</option>
                <option value="1">1-Car Garage</option>
                <option value="2" selected>2-Car Garage</option>
                <option value="3">3-Car Garage</option>
              </select>
            </div>
            <div>
              <div class="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                <label for="quality"><i class="fa-solid fa-star text-amber-500 mr-1"></i> Neighborhood Quality</label>
                <span id="quality-val" class="font-mono text-amber-600 font-bold">7 / 10</span>
              </div>
              <input type="range" id="quality" min="1" max="10" step="1" value="7"
                class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
                oninput="syncVal('quality', 'quality-val', ' / 10', false); submitPrediction();">
            </div>
          </div>

          <button type="submit" class="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-bold shadow-md hover:shadow-lg transition flex items-center justify-center gap-2">
            <i class="fa-solid fa-calculator"></i> Calculate Property Valuation
          </button>
        </form>
      </div>

      <!-- Result Card & Feature Breakdown (5 cols) -->
      <div class="lg:col-span-5 flex flex-col gap-6">
        
        <!-- Big Price Card -->
        <div class="bg-gradient-to-br from-emerald-600 to-teal-700 text-white p-6 rounded-2xl shadow-lg relative overflow-hidden">
          <div class="absolute -right-6 -bottom-6 opacity-10 text-9xl">
            <i class="fa-solid fa-dollar-sign"></i>
          </div>
          <div class="text-xs uppercase tracking-wider font-semibold text-emerald-100">Estimated Market Valuation</div>
          <div id="display-price" class="text-4xl sm:text-5xl font-black font-mono tracking-tight mt-2 mb-1">$630,547.73</div>
          <div class="text-xs text-emerald-100 flex items-center gap-1.5 mt-3">
            <i class="fa-solid fa-circle-check"></i>
            <span>Predicted via Scikit-Learn Linear Regression Pipeline</span>
          </div>
        </div>

        <!-- Linear Equation Decomposition -->
        <div class="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 flex-1 flex flex-col">
          <h3 class="text-sm font-bold text-slate-900 mb-3 flex items-center gap-2">
            <i class="fa-solid fa-chart-pie text-blue-600"></i> Mathematical Impact Breakdown
          </h3>
          
          <div class="overflow-y-auto max-h-64 flex-1 pr-1 space-y-2 text-xs">
            <div class="flex justify-between py-1.5 border-b border-slate-100">
              <span class="text-slate-500 font-medium">Model Intercept (β₀)</span>
              <span class="font-mono font-bold text-slate-700">-$2,010,567.42</span>
            </div>
            <div class="flex justify-between py-1.5 border-b border-slate-100">
              <span class="text-slate-600">Square Footage (<span id="bd-sqft">2800</span> × $199.51)</span>
              <span id="impact-sqft" class="font-mono font-bold text-emerald-600">+$558,636.96</span>
            </div>
            <div class="flex justify-between py-1.5 border-b border-slate-100">
              <span class="text-slate-600">Bedrooms (<span id="bd-bed">3</span> × $10,225.20)</span>
              <span id="impact-bed" class="font-mono font-bold text-emerald-600">+$30,675.61</span>
            </div>
            <div class="flex justify-between py-1.5 border-b border-slate-100">
              <span class="text-slate-600">Bathrooms (<span id="bd-bath">2</span> × $8,208.43)</span>
              <span id="impact-bath" class="font-mono font-bold text-emerald-600">+$16,416.87</span>
            </div>
            <div class="flex justify-between py-1.5 border-b border-slate-100">
              <span class="text-slate-600">Year Built (<span id="bd-year">2000</span> × $993.54)</span>
              <span id="impact-year" class="font-mono font-bold text-emerald-600">+$1,987,074.40</span>
            </div>
            <div class="flex justify-between py-1.5 border-b border-slate-100">
              <span class="text-slate-600">Lot Size (<span id="bd-lot">2.5</span> × $14,885.38)</span>
              <span id="impact-lot" class="font-mono font-bold text-emerald-600">+$37,213.46</span>
            </div>
            <div class="flex justify-between py-1.5 border-b border-slate-100">
              <span class="text-slate-600">Garage Size (<span id="bd-gar">2</span> × $5,146.15)</span>
              <span id="impact-gar" class="font-mono font-bold text-emerald-600">+$10,292.30</span>
            </div>
            <div class="flex justify-between py-1.5">
              <span class="text-slate-600">Quality Rating (<span id="bd-qual">7</span> × $115.07)</span>
              <span id="impact-qual" class="font-mono font-bold text-emerald-600">+$805.48</span>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- Visualizations Showcase Gallery -->
    <div class="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-4 mb-6 border-b border-slate-100 gap-4">
        <div>
          <h2 class="text-lg font-bold text-slate-900">Model Visualizations & EDA Gallery</h2>
          <p class="text-xs text-slate-500">Publication-quality 300 DPI analytical charts generated during model evaluation</p>
        </div>
        
        <!-- Tab selector -->
        <div class="flex flex-wrap gap-2 text-xs">
          <button onclick="showPlot('actual')" class="plot-btn px-3 py-1.5 rounded-lg bg-blue-600 text-white font-semibold">Actual vs Predicted</button>
          <button onclick="showPlot('corr')" class="plot-btn px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200 font-semibold">Correlation Heatmap</button>
          <button onclick="showPlot('features')" class="plot-btn px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200 font-semibold">Feature Fits</button>
          <button onclick="showPlot('dist')" class="plot-btn px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200 font-semibold">Distribution</button>
          <button onclick="showPlot('residual')" class="plot-btn px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200 font-semibold">Residual Plot</button>
        </div>
      </div>

      <!-- Plot Viewer Container -->
      <div class="text-center bg-slate-50 rounded-xl p-4 border border-slate-200">
        <img id="active-plot" src="/visualizations/actual_vs_predicted.png" alt="Visualization Plot" class="max-h-[500px] mx-auto rounded-lg shadow-sm object-contain">
        <p id="plot-caption" class="mt-4 text-xs font-medium text-slate-600">
          <strong>Key Academic Finding:</strong> Test points adhere tightly along the red y = x diagonal, validating R² = 99.84%.
        </p>
      </div>
    </div>

  </main>

  <footer class="bg-white border-t border-slate-200 py-4 text-center text-xs text-slate-500">
    House Price Prediction — Academic Machine Learning Demonstration Project | Linear Regression
  </footer>

  <script>
    const beta0 = -2010567.4203;
    const weights = {
      sqft: 199.5132,
      bed: 10225.2044,
      bath: 8208.4348,
      year: 993.5372,
      lot: 14885.3844,
      gar: 5146.1484,
      qual: 115.0686
    };

    function syncVal(inputId, labelId, suffix, isNum) {
      const val = document.getElementById(inputId).value;
      const formatted = isNum ? Number(val).toLocaleString() : val;
      document.getElementById(labelId).textContent = formatted + suffix;
    }

    function resetDefaults() {
      document.getElementById('sqft').value = 2800;
      document.getElementById('bed').value = 3;
      document.getElementById('bath').value = 2;
      document.getElementById('year').value = 2000;
      document.getElementById('lot').value = 2.5;
      document.getElementById('garage').value = 2;
      document.getElementById('quality').value = 7;
      syncVal('sqft', 'sqft-val', ' sq ft', true);
      syncVal('quality', 'quality-val', ' / 10', false);
      submitPrediction();
    }

    function submitPrediction() {
      const sqft = parseFloat(document.getElementById('sqft').value) || 0;
      const bed = parseFloat(document.getElementById('bed').value) || 0;
      const bath = parseFloat(document.getElementById('bath').value) || 0;
      const year = parseFloat(document.getElementById('year').value) || 0;
      const lot = parseFloat(document.getElementById('lot').value) || 0;
      const gar = parseFloat(document.getElementById('garage').value) || 0;
      const qual = parseFloat(document.getElementById('quality').value) || 0;

      const imp_sqft = sqft * weights.sqft;
      const imp_bed = bed * weights.bed;
      const imp_bath = bath * weights.bath;
      const imp_year = year * weights.year;
      const imp_lot = lot * weights.lot;
      const imp_gar = gar * weights.gar;
      const imp_qual = qual * weights.qual;

      const price = beta0 + imp_sqft + imp_bed + imp_bath + imp_year + imp_lot + imp_gar + imp_qual;

      document.getElementById('display-price').textContent = '$' + price.toLocaleString('en-US', {
        minimumFractionDigits: 2, maximumFractionDigits: 2
      });

      document.getElementById('bd-sqft').textContent = sqft.toLocaleString();
      document.getElementById('bd-bed').textContent = bed;
      document.getElementById('bd-bath').textContent = bath;
      document.getElementById('bd-year').textContent = year;
      document.getElementById('bd-lot').textContent = lot;
      document.getElementById('bd-gar').textContent = gar;
      document.getElementById('bd-qual').textContent = qual;

      document.getElementById('impact-sqft').textContent = '+$' + imp_sqft.toLocaleString('en-US', {maximumFractionDigits: 2});
      document.getElementById('impact-bed').textContent = '+$' + imp_bed.toLocaleString('en-US', {maximumFractionDigits: 2});
      document.getElementById('impact-bath').textContent = '+$' + imp_bath.toLocaleString('en-US', {maximumFractionDigits: 2});
      document.getElementById('impact-year').textContent = '+$' + imp_year.toLocaleString('en-US', {maximumFractionDigits: 2});
      document.getElementById('impact-lot').textContent = '+$' + imp_lot.toLocaleString('en-US', {maximumFractionDigits: 2});
      document.getElementById('impact-gar').textContent = '+$' + imp_gar.toLocaleString('en-US', {maximumFractionDigits: 2});
      document.getElementById('impact-qual').textContent = '+$' + imp_qual.toLocaleString('en-US', {maximumFractionDigits: 2});
    }

    const plots = {
      'actual': {
        src: '/visualizations/actual_vs_predicted.png',
        caption: '<strong>Key Academic Finding:</strong> Test points adhere tightly along the red y = x diagonal, validating R² = 99.84%.'
      },
      'corr': {
        src: '/visualizations/correlation_heatmap.png',
        caption: '<strong>Key Academic Finding:</strong> Square Footage demonstrates the strongest linear correlation (r ≈ 0.99) with House Price.'
      },
      'features': {
        src: '/visualizations/feature_vs_house_price.png',
        caption: '<strong>Key Academic Finding:</strong> Multi-panel regression subplots displaying individual linear feature trends.'
      },
      'dist': {
        src: '/visualizations/house_price_distribution.png',
        caption: '<strong>Key Academic Finding:</strong> Bell-shaped normal distribution profile with Mean = $618,861 and Median = $628,267.'
      },
      'residual': {
        src: '/visualizations/residual_plot.png',
        caption: '<strong>Key Academic Finding:</strong> Homoscedasticity confirmed: errors randomly scattered without cone or funneling.'
      }
    };

    function showPlot(key) {
      document.querySelectorAll('.plot-btn').forEach(b => {
        b.classList.remove('bg-blue-600', 'text-white');
        b.classList.add('bg-slate-100', 'text-slate-700');
      });
      event.target.classList.remove('bg-slate-100', 'text-slate-700');
      event.target.classList.add('bg-blue-600', 'text-white');

      const plot = plots[key];
      if (plot) {
        document.getElementById('active-plot').src = plot.src;
        document.getElementById('plot-caption').innerHTML = plot.caption;
      }
    }
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        path = url.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
            return

        if path.startswith("/visualizations/"):
            filename = os.path.basename(path)
            filepath = os.path.join(VIS_DIR, filename)
            if os.path.isfile(filepath) and filename.endswith(".png"):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return

        if path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(METADATA).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
                input_df = pd.DataFrame([{
                    "Square_Footage": float(data.get("sqft", 2800)),
                    "Num_Bedrooms": int(data.get("bed", 3)),
                    "Num_Bathrooms": int(data.get("bath", 2)),
                    "Year_Built": int(data.get("year", 2000)),
                    "Lot_Size": float(data.get("lot", 2.5)),
                    "Garage_Size": int(data.get("garage", 2)),
                    "Neighborhood_Quality": int(data.get("quality", 7)),
                }])
                price = predict_price(PIPELINE, input_df)
                resp = {"price": price, "formatted": f"${price:,.2f}"}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def launch_web_server(port=5000, open_browser=True):
    load_globals()
    server_address = ("127.0.0.1", port)
    
    # Try finding an open port if 5000 is occupied
    httpd = None
    for p in range(port, port + 10):
        try:
            httpd = HTTPServer(("127.0.0.1", p), DashboardHandler)
            port = p
            break
        except OSError:
            continue

    if httpd is None:
        print("[ERROR] Could not bind to any port.")
        return

    url = f"http://127.0.0.1:{port}"
    print(f"\n=======================================================")
    print(f" House Price Prediction — Web Dashboard Running")
    print(f" Local URL: {url}")
    print(f" Press Ctrl+C in terminal to stop the web server")
    print(f"=======================================================\n")

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web dashboard server...")
        httpd.server_close()


if __name__ == "__main__":
    launch_web_server()

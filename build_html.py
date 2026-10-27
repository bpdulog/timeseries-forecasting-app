import json
from pathlib import Path

root = Path(__file__).resolve().parent
with open(root / 'app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

with open(root / 'sample_data.csv', 'r', encoding='utf-8') as f:
    csv_code = f.read()

files_dict = {
    'app.py': app_code,
    'sample_data.csv': csv_code
}

files_json = json.dumps(files_dict)

html_template = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <title>Time Series Forecasting & Diagnostic Suite</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.58.0/build/stlite.css" />
    <style>
      body, html {
        margin: 0;
        padding: 0;
        height: 100%;
        overflow: hidden;
        background-color: #f8f9fa;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }
      #loading-screen {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
        color: #2b3a42;
        text-align: center;
        padding: 20px;
      }
      .spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #e0e0e0;
        border-top: 5px solid #1f77b4;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 20px;
      }
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    </style>
  </head>
  <body>
    <div id="root">
      <div id="loading-screen">
        <div class="spinner"></div>
        <h2>Initializing Pyodide WebAssembly Environment...</h2>
        <p>Loading Python runtime, pandas, statsmodels, and econometric forecasting engine directly in your browser.</p>
        <p style="color: #6c757d; font-size: 0.9em;">This initial load takes ~10-15 seconds on first visit (files are cached afterwards).</p>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.58.0/build/stlite.js"></script>
    <script>
      const files = __FILES_PLACEHOLDER__;
      stlite.mount(
        {
          requirements: ["pandas", "numpy", "statsmodels", "scipy", "plotly==5.24.1", "openpyxl"],
          entrypoint: "app.py",
          files: files,
        },
        document.getElementById("root")
      );
    </script>
  </body>
</html>
"""

html_final = html_template.replace("__FILES_PLACEHOLDER__", files_json)

with open(root / 'index.html', 'w', encoding='utf-8') as f:
    f.write(html_final)

print(f"Generated index.html successfully ({len(html_final)} bytes)")

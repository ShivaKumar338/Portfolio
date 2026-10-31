import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000

# Change working directory to directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

Handler = http.server.SimpleHTTPRequestHandler

class CustomHandler(Handler):
    def end_headers(self):
        # Enable caching-free local development
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

print(f"============================================================")
print(f"🚀 Shiva Kumar Madagoni - Portfolio Server Running!")
print(f"📡 Local URL: http://localhost:{PORT}")
print(f"📂 Serving directory: {script_dir}")
print(f"Press Ctrl+C to terminate.")
print(f"============================================================")

try:
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        try:
            webbrowser.open(f"http://localhost:{PORT}")
        except Exception:
            pass
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped gracefully.")
except OSError as e:
    print(f"\nPort {PORT} already in use or error: {e}")
    print(f"You can open file://{os.path.abspath('index.html')} directly in your browser!")

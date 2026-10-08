from http.server import SimpleHTTPRequestHandler, HTTPServer
class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
HTTPServer(('0.0.0.0', 8000), CORSRequestHandler).serve_forever()

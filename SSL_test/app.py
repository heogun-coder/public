from flask import Flask, render_template, jsonify, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/data", methods=["GET", "POST"])
def api_data():
    if request.method == "POST":
        data = request.json
        print("Received data:", data)
        return jsonify({"status": "success", "message": "Data received"}), 200
    return jsonify({"message": "Send a POST request with JSON data."}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, ssl_context=("cert.pem", "key.pem"))

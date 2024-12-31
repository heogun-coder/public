from flask import Flask, request, jsonify, render_template
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/send", methods=["POST"])
def send():
    try:
        # 클라이언트에서 받은 데이터
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400

        # jsonplaceholder로 데이터 전송
        response = requests.post(
            "https://jsonplaceholder.typicode.com/posts", json=data
        )
        if response.status_code == 201:
            return (
                jsonify(
                    {
                        "message": "Data successfully sent to JSONPlaceholder!",
                        "data_sent": data,
                        "server_response": response.json(),
                    }
                ),
                201,
            )
        else:
            return (
                jsonify({"error": "Failed to send data", "details": response.text}),
                response.status_code,
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="192.168.0.17", port=5000)

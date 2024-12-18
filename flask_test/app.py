from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)


# 루트 페이지
@app.route("/")
def home():
    return render_template("index.html")


# 외부 서버에 GET 요청 보내기
@app.route("/fetch-data", methods=["GET"])
def fetch_data():
    external_url = "https://jsonplaceholder.typicode.com/posts/1"
    try:
        response = requests.get(external_url)
        response.raise_for_status()
        return jsonify(
            {"message": "Data fetched successfully", "data": response.json()}
        )
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500


# 외부 서버에 POST 요청 보내기
@app.route("/send-data", methods=["POST"])
def send_data():
    external_url = "https://jsonplaceholder.typicode.com/posts"
    input_data = request.get_json()
    try:
        response = requests.post(external_url, json=input_data)
        response.raise_for_status()
        return jsonify(
            {"message": "Data sent successfully", "response": response.json()}
        )
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

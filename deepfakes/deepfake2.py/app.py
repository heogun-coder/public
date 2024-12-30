from flask import Flask, render_template, request, jsonify
import os
from face_swap import process_videos

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/swap", methods=["POST"])
def swap_faces():
    if "source" not in request.files or "target" not in request.files:
        return jsonify({"error": "비디오 파일이 필요합니다"}), 400

    source_video = request.files["source"]
    target_video = request.files["target"]

    # 파일 저장
    source_path = os.path.join(UPLOAD_FOLDER, "source.mp4")
    target_path = os.path.join(UPLOAD_FOLDER, "target.mp4")
    source_video.save(source_path)
    target_video.save(target_path)

    # 얼굴 스왑 처리
    output_path = os.path.join(RESULT_FOLDER, "result.mp4")
    process_videos(source_path, target_path, output_path)

    return jsonify({"result": "/static/results/result.mp4"})


if __name__ == "__main__":
    app.run(debug=True)

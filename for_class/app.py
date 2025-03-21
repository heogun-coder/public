from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "your_secret_key"  # 보안을 위해 실제 배포시 더 강력한 값 사용

# 데모를 위한 메모리 내 저장소 (실제 운영에서는 데이터베이스 사용 권장)
calendar_events = []
lab_reservations = []


@app.route("/")
def index():
    # 기본 페이지로 달력 페이지로 리디렉션
    return redirect(url_for("calendar_view"))


@app.route("/calendar", methods=["GET", "POST"])
def calendar_view():
    if request.method == "POST":
        event_title = request.form.get("event_title")
        event_date = request.form.get("event_date")
        if event_title and event_date:
            # 새로운 이벤트를 리스트에 추가
            calendar_events.append({"title": event_title, "date": event_date})
            flash("이벤트가 성공적으로 추가되었습니다.", "success")
            return redirect(url_for("calendar_view"))
        else:
            flash("이벤트 제목과 날짜를 모두 입력하세요.", "error")
    return render_template("calendar.html", events=calendar_events)


@app.route("/reservation", methods=["GET", "POST"])
def reservation():
    if request.method == "POST":
        lab_name = request.form.get("lab_name")
        reserved_date = request.form.get("reserved_date")
        reserved_time = request.form.get("reserved_time")
        user_name = request.form.get("user_name")
        if lab_name and reserved_date and reserved_time and user_name:
            # 예약 정보를 리스트에 추가
            lab_reservations.append(
                {
                    "lab_name": lab_name,
                    "reserved_date": reserved_date,
                    "reserved_time": reserved_time,
                    "user_name": user_name,
                }
            )
            flash("예약이 성공적으로 완료되었습니다.", "success")
            return redirect(url_for("reservation"))
        else:
            flash("모든 필드를 입력하세요.", "error")
    return render_template("reservation.html", reservations=lab_reservations)


if __name__ == "__main__":
    app.run(debug=True)

import schedule
import time
import subprocess
import sys
from datetime import datetime
import pytz


def run_todo_app():
    # Python 실행 파일 경로와 to_do_list.py 스크립트 경로
    python_path = sys.executable
    script_path = "to_do_list.py"

    # 서브프로세스로 실행
    subprocess.Popen([python_path, script_path])


def main():
    # 한국 시간대 설정
    korea_tz = pytz.timezone("Asia/Seoul")

    # 매일 아침 8시에 실행되도록 스케줄 설정
    schedule.every().day.at("08:00").do(run_todo_app)

    while True:
        schedule.run_pending()
        time.sleep(60)  # 1분마다 체크


if __name__ == "__main__":
    main()

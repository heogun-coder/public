import tkinter as tk
from tkinter import ttk
from datetime import datetime
import json


class TodoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("과목별 할 일 관리")

        # 데이터 구조
        self.tasks = []

        # 메인 프레임
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 입력 섹션
        self.create_input_section()

        # 표시 섹션
        self.create_display_section()

        # 저장된 데이터 로드
        self.load_tasks()

        # 삭제/수정 버튼 추가
        self.create_control_buttons()

        # 트리뷰 선택 이벤트 바인딩
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        self.selected_item = None  # 선택된 항목 저장용
        self.is_editing = False  # 수정 모드 플래그

    def create_input_section(self):
        # 과목 선택
        subjects = ["수학", "물리", "화학", "생명과학", "지구과학"]
        ttk.Label(self.main_frame, text="과목:").grid(row=0, column=0, sticky=tk.W)
        self.subject_var = tk.StringVar()
        self.subject_combo = ttk.Combobox(
            self.main_frame, textvariable=self.subject_var, values=subjects
        )
        self.subject_combo.grid(row=0, column=1, sticky=tk.W)

        # 문제집 입력
        ttk.Label(self.main_frame, text="문제집:").grid(row=1, column=0, sticky=tk.W)
        self.workbook_var = tk.StringVar()
        ttk.Entry(self.main_frame, textvariable=self.workbook_var).grid(
            row=1, column=1, sticky=tk.W
        )

        # 단원 입력
        ttk.Label(self.main_frame, text="단원:").grid(row=2, column=0, sticky=tk.W)
        self.chapter_var = tk.StringVar()
        ttk.Entry(self.main_frame, textvariable=self.chapter_var).grid(
            row=2, column=1, sticky=tk.W
        )

        # 범위 입력
        ttk.Label(self.main_frame, text="범위:").grid(row=3, column=0, sticky=tk.W)
        self.range_var = tk.StringVar()
        ttk.Entry(self.main_frame, textvariable=self.range_var).grid(
            row=3, column=1, sticky=tk.W
        )

        # Due date 입력
        ttk.Label(self.main_frame, text="마감일(YYYY-MM-DD):").grid(
            row=4, column=0, sticky=tk.W
        )
        self.due_date_var = tk.StringVar()
        ttk.Entry(self.main_frame, textvariable=self.due_date_var).grid(
            row=4, column=1, sticky=tk.W
        )

        # 추가 버튼
        ttk.Button(self.main_frame, text="추가", command=self.add_task).grid(
            row=5, column=0, columnspan=2
        )

    def create_display_section(self):
        # 표시 옵션
        ttk.Label(self.main_frame, text="정렬 기준:").grid(row=6, column=0, sticky=tk.W)
        self.sort_var = tk.StringVar(value="과목별")
        ttk.Radiobutton(
            self.main_frame,
            text="과목별",
            variable=self.sort_var,
            value="과목별",
            command=self.update_display,
        ).grid(row=6, column=1, sticky=tk.W)
        ttk.Radiobutton(
            self.main_frame,
            text="마감일별",
            variable=self.sort_var,
            value="마감일별",
            command=self.update_display,
        ).grid(row=6, column=2, sticky=tk.W)

        # 할 일 목록 표시
        self.tree = ttk.Treeview(
            self.main_frame,
            columns=("과목", "문제집", "단원", "범위", "마감일"),
            show="headings",
        )
        self.tree.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 컬럼 설정
        for col in ("과목", "문제집", "단원", "범위", "마감일"):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

    def create_control_buttons(self):
        # 버튼 프레임
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=5)

        # 추가 버튼
        self.add_button = ttk.Button(button_frame, text="추가", command=self.add_task)
        self.add_button.grid(row=0, column=0, padx=5)

        # 수정 버튼
        self.edit_button = ttk.Button(button_frame, text="수정", command=self.edit_task)
        self.edit_button.grid(row=0, column=1, padx=5)

        # 삭제 버튼
        self.delete_button = ttk.Button(
            button_frame, text="삭제", command=self.delete_task
        )
        self.delete_button.grid(row=0, column=2, padx=5)

    def on_select(self, event):
        selected_items = self.tree.selection()
        if selected_items:
            self.selected_item = selected_items[0]
            # 선택된 항목의 값을 입력 필드에 표시
            values = self.tree.item(self.selected_item)["values"]
            if not self.is_editing:  # 수정 중이 아닐 때만 값을 설정
                self.subject_var.set(values[0])
                self.workbook_var.set(values[1])
                self.chapter_var.set(values[2])
                self.range_var.set(values[3])
                self.due_date_var.set(values[4])

    def edit_task(self):
        if not self.selected_item:
            return

        if not self.is_editing:
            # 수정 모드 시작
            self.is_editing = True
            self.add_button.configure(text="저장")
        else:
            # 수정 사항 저장
            self.is_editing = False
            self.add_button.configure(text="추가")

            # 선택된 항목의 인덱스 찾기
            values = self.tree.item(self.selected_item)["values"]
            task_index = next(
                i
                for i, task in enumerate(self.tasks)
                if task["과목"] == values[0]
                and task["문제집"] == values[1]
                and task["단원"] == values[2]
                and task["범위"] == values[3]
                and task["마감일"] == values[4]
            )

            # 수정된 데이터로 업데이트
            self.tasks[task_index] = {
                "과목": self.subject_var.get(),
                "문제집": self.workbook_var.get(),
                "단원": self.chapter_var.get(),
                "범위": self.range_var.get(),
                "마감일": self.due_date_var.get(),
            }

            self.save_tasks()
            self.update_display()
            self.clear_inputs()
            self.selected_item = None

    def delete_task(self):
        if not self.selected_item:
            return

        # 선택된 항목의 값
        values = self.tree.item(self.selected_item)["values"]

        # tasks 리스트에서 해당 항목 찾아 삭제
        self.tasks = [
            task
            for task in self.tasks
            if not (
                task["과목"] == values[0]
                and task["문제집"] == values[1]
                and task["단원"] == values[2]
                and task["범위"] == values[3]
                and task["마감일"] == values[4]
            )
        ]

        self.save_tasks()
        self.update_display()
        self.clear_inputs()
        self.selected_item = None

    def add_task(self):
        if self.is_editing:
            self.edit_task()
            return

        task = {
            "과목": self.subject_var.get(),
            "문제집": self.workbook_var.get(),
            "단원": self.chapter_var.get(),
            "범위": self.range_var.get(),
            "마감일": self.due_date_var.get(),
        }

        self.tasks.append(task)
        self.save_tasks()
        self.update_display()
        self.clear_inputs()

    def clear_inputs(self):
        self.subject_var.set("")
        self.workbook_var.set("")
        self.chapter_var.set("")
        self.range_var.set("")
        self.due_date_var.set("")

    def update_display(self):
        # 기존 항목 삭제
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 정렬된 할 일 목록
        sorted_tasks = self.sort_tasks()

        # 새로운 항목 추가
        for task in sorted_tasks:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    task["과목"],
                    task["문제집"],
                    task["단원"],
                    task["범위"],
                    task["마감일"],
                ),
            )

    def sort_tasks(self):
        if self.sort_var.get() == "과목별":
            return sorted(self.tasks, key=lambda x: x["과목"])
        else:  # 마감일별
            return sorted(
                self.tasks, key=lambda x: datetime.strptime(x["마감일"], "%Y-%m-%d")
            )

    def save_tasks(self):
        with open("tasks.json", "w", encoding="utf-8") as f:
            json.dump(self.tasks, f, ensure_ascii=False, indent=2)

    def load_tasks(self):
        try:
            with open("tasks.json", "r", encoding="utf-8") as f:
                self.tasks = json.load(f)
            self.update_display()
        except FileNotFoundError:
            self.tasks = []


if __name__ == "__main__":
    root = tk.Tk()
    app = TodoApp(root)
    root.mainloop()

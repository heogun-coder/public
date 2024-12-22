import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import re


class PDFEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("PDF Editor")
        self.master.geometry("500x500")

        self.pdf_path = ""
        self.total_pages = 0
        self.pdf_files = []
        self.transformations = {}

        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self.master)
        notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # Slicer tab
        slicer_frame = ttk.Frame(notebook)
        notebook.add(slicer_frame, text="PDF Slicer")
        self.create_slicer_widgets(slicer_frame)

        # Connector tab
        connector_frame = ttk.Frame(notebook)
        notebook.add(connector_frame, text="PDF Connector")
        self.create_connector_widgets(connector_frame)

        # Transformer tab
        transformer_frame = ttk.Frame(notebook)
        notebook.add(transformer_frame, text="Word Transformer")
        self.create_transformer_widgets(transformer_frame)

    def create_slicer_widgets(self, parent):
        # File selection
        self.select_button = ttk.Button(
            parent, text="Select PDF", command=self.select_pdf
        )
        self.select_button.pack(pady=10)

        self.file_label = ttk.Label(parent, text="No file selected")
        self.file_label.pack()

        # Page range input
        self.range_frame = ttk.Frame(parent)
        self.range_frame.pack(pady=10)

        ttk.Label(self.range_frame, text="Start Page:").grid(row=0, column=0)
        self.start_page = ttk.Entry(self.range_frame, width=5)
        self.start_page.grid(row=0, column=1)

        ttk.Label(self.range_frame, text="End Page:").grid(row=0, column=2)
        self.end_page = ttk.Entry(self.range_frame, width=5)
        self.end_page.grid(row=0, column=3)

        # Slice button
        self.slice_button = ttk.Button(parent, text="Slice PDF", command=self.slice_pdf)
        self.slice_button.pack(pady=10)

    def create_connector_widgets(self, parent):
        # File selection
        self.add_button = ttk.Button(parent, text="Add PDF", command=self.add_pdf)
        self.add_button.pack(pady=10)

        # List of selected PDFs
        self.file_listbox = tk.Listbox(parent, width=50)
        self.file_listbox.pack(pady=10)

        # Remove selected PDF
        self.remove_button = ttk.Button(
            parent, text="Remove Selected", command=self.remove_pdf
        )
        self.remove_button.pack(pady=5)

        # Connect PDFs button
        self.connect_button = ttk.Button(
            parent, text="Connect PDFs", command=self.connect_pdfs
        )
        self.connect_button.pack(pady=10)

    def create_transformer_widgets(self, parent):
        # File selection
        self.trans_select_button = ttk.Button(
            parent, text="Select PDF", command=self.select_pdf_for_transformation
        )
        self.trans_select_button.pack(pady=10)

        self.trans_file_label = ttk.Label(parent, text="No file selected")
        self.trans_file_label.pack()

        # Word transformation input
        self.trans_frame = ttk.Frame(parent)
        self.trans_frame.pack(pady=10)

        ttk.Label(self.trans_frame, text="Original Word:").grid(row=0, column=0)
        self.original_word = ttk.Entry(self.trans_frame, width=20)
        self.original_word.grid(row=0, column=1)

        ttk.Label(self.trans_frame, text="Transform to:").grid(row=1, column=0)
        self.transform_to = ttk.Entry(self.trans_frame, width=20)
        self.transform_to.grid(row=1, column=1)

        # Add transformation button
        self.add_trans_button = ttk.Button(
            parent, text="Add Transformation", command=self.add_transformation
        )
        self.add_trans_button.pack(pady=5)

        # Transformation list
        self.trans_listbox = tk.Listbox(parent, width=50)
        self.trans_listbox.pack(pady=10)

        # Remove transformation button
        self.remove_trans_button = ttk.Button(
            parent, text="Remove Transformation", command=self.remove_transformation
        )
        self.remove_trans_button.pack(pady=5)

        # Apply transformations button
        self.apply_trans_button = ttk.Button(
            parent, text="Apply Transformations", command=self.apply_transformations
        )
        self.apply_trans_button.pack(pady=10)

    def select_pdf(self):
        self.pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if self.pdf_path:
            filename = os.path.basename(self.pdf_path)
            self.file_label.config(text=f"Selected: {filename}")

            with open(self.pdf_path, "rb") as file:
                pdf = PdfReader(file)
                self.total_pages = len(pdf.pages)

    def slice_pdf(self):
        if not self.pdf_path:
            messagebox.showerror("Error", "Please select a PDF file first")
            return

        try:
            start = int(self.start_page.get())
            end = int(self.end_page.get())

            if start < 1 or end > self.total_pages or start > end:
                raise ValueError

            pdf_writer = PdfWriter()
            pdf_reader = PdfReader(self.pdf_path)

            for page in range(start - 1, end):
                pdf_writer.add_page(pdf_reader.pages[page])

            output_path = filedialog.asksaveasfilename(
                defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")]
            )
            if output_path:
                with open(output_path, "wb") as output_file:
                    pdf_writer.write(output_file)
                messagebox.showinfo("Success", "PDF sliced successfully!")

        except ValueError:
            messagebox.showerror(
                "Error", f"Please enter valid page numbers (1-{self.total_pages})"
            )

    def add_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.pdf_files.append(file_path)
            self.file_listbox.insert(tk.END, os.path.basename(file_path))

    def remove_pdf(self):
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self.file_listbox.delete(index)
            self.pdf_files.pop(index)

    def connect_pdfs(self):
        if len(self.pdf_files) < 2:
            messagebox.showerror(
                "Error", "Please select at least two PDF files to connect"
            )
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")]
        )
        if output_path:
            pdf_writer = PdfWriter()

            for pdf_file in self.pdf_files:
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    pdf_writer.add_page(page)

            with open(output_path, "wb") as output_file:
                pdf_writer.write(output_file)

            messagebox.showinfo("Success", "PDFs connected successfully!")

    def select_pdf_for_transformation(self):
        self.pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if self.pdf_path:
            filename = os.path.basename(self.pdf_path)
            self.trans_file_label.config(text=f"Selected: {filename}")

    def add_transformation(self):
        original = self.original_word.get().strip()
        transform = self.transform_to.get().strip()
        if original and transform:
            self.transformations[original] = transform
            self.trans_listbox.insert(tk.END, f"{original} → {transform}")
            self.original_word.delete(0, tk.END)
            self.transform_to.delete(0, tk.END)
        else:
            messagebox.showwarning(
                "Warning", "Please enter both original word and its transformation."
            )

    def remove_transformation(self):
        selection = self.trans_listbox.curselection()
        if selection:
            index = selection[0]
            item = self.trans_listbox.get(index)
            original = item.split(" → ")[0]
            del self.transformations[original]
            self.trans_listbox.delete(index)

    def apply_transformations(self):
        if not self.pdf_path:
            messagebox.showerror("Error", "Please select a PDF file first")
            return

        if not self.transformations:
            messagebox.showwarning("Warning", "No transformations to apply")
            return

        try:
            pdf_reader = PdfReader(self.pdf_path)
            pdf_writer = PdfWriter()

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                content = page.extract_text()

                for original, transform in self.transformations.items():
                    # Use word boundaries to ensure we're replacing whole words
                    pattern = r"\b" + re.escape(original) + r"\b"
                    content = re.sub(pattern, transform, content)

                # Create a new PDF page with the transformed content
                packet = BytesIO()
                can = canvas.Canvas(packet, pagesize=letter)
                can.drawString(100, 700, content)  # Adjust position as needed
                can.save()

                packet.seek(0)
                new_page = PdfReader(packet).pages[0]
                pdf_writer.add_page(new_page)

            output_path = filedialog.asksaveasfilename(
                defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")]
            )
            if output_path:
                with open(output_path, "wb") as output_file:
                    pdf_writer.write(output_file)
                messagebox.showinfo("Success", "Transformations applied successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PDFEditor(root)
    root.mainloop()
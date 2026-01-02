
from fpdf import FPDF
import pandas as pd
import tempfile
import plotly.io as pio
import os

class ReportGenerator(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", size=12)

    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "EquiScan: Algorithmic Ethics Audit Report", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_section_title(self, title):
        self.set_font("Arial", "B", 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(5)

    def add_text(self, text):
        self.set_font("Arial", size=12)
        self.multi_cell(0, 10, text)
        self.ln(5)
        
    def add_metric(self, name, value, status=None):
        self.set_font("Arial", "B", 12)
        text = f"{name}: {value}"
        if status:
            text += f" ({status})"
        self.cell(0, 10, text, ln=True)
        self.set_font("Arial", size=12)

    def add_plot(self, fig, title="Plot"):
        self.add_section_title(title)
        
        # Save Plotly figure to temporary image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            # Requires kaleido
            try:
                pio.write_image(fig, tmp_file.name, format="png", width=800, height=500)
                self.image(tmp_file.name, w=170)
                self.ln(10)
            except Exception as e:
                self.add_text(f"[Error generating plot: {e}]")
            finally:
                # Cleanup happens after image is added? 
                # FPDF reads file immediately? Yes.
                pass
                
        # We should delete the temp file, but FPDF might need it until output?
        # Usually fine to delete after adding.
        try:
            os.remove(tmp_file.name)
        except:
            pass

    def save_report(self, filename="audit_report.pdf"):
        return self.output(dest="S").encode("latin-1") # Return bytes for download

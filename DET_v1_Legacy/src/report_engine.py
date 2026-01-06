from fpdf import FPDF
import pandas as pd
import tempfile
import plotly.io as pio
import os
from typing import Dict, Any

class ReportGenerator(FPDF):
    """
    DET v3 Dedicated Report Generator.
    Focuses exclusively on dataset ethical triage results.
    """
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", size=12)

    def header(self):
        # Header with branding
        self.set_font("Arial", "B", 18)
        self.set_text_color(15, 23, 42) # Slate 900
        self.cell(0, 15, "EquiScan DET v3: Ethical Triage Report", ln=True, align="C")
        self.set_text_color(0, 0, 0)
        self.line(10, 25, 200, 25)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Confidential Audit Report - Page {self.page_no()}", align="C")

    def add_section_title(self, title):
        self.set_font("Arial", "B", 14)
        self.set_fill_color(241, 245, 249) # Slate 50
        self.cell(0, 10, f"  {title}", ln=True, fill=True)
        self.ln(5)

    def add_text(self, text, size=12, style=''):
        self.set_font("Arial", style=style, size=size)
        self.multi_cell(0, 8, text)
        self.ln(3)
        
    def add_metric_row(self, name, score, flag):
        self.set_font("Arial", "B", 11)
        self.cell(100, 10, name, border=1)
        
        # Color code flag using Pro Dark Palette
        if flag == 'RED':
            self.set_fill_color(248, 113, 113) # Red 400
        elif flag == 'YELLOW':
            self.set_fill_color(250, 204, 21) # Yellow 400
        else:
            self.set_fill_color(74, 222, 128) # Green 400
            
        self.cell(30, 10, f" {score:.3f}" if isinstance(score, float) else f" {score}", border=1)
        self.cell(60, 10, f" {flag}", border=1, fill=True)
        self.ln()

    def add_det_summary(self, decision_result: Dict[str, Any]):
        self.add_section_title("Executive Triage Summary")
        
        decision = decision_result['decision']
        confidence = decision_result['confidence']
        
        # Big Decision Box
        self.set_font("Arial", "B", 16)
        if decision == 'PROCEED':
            self.set_text_color(22, 163, 74) # Green 600
        elif decision == 'MITIGATE':
            self.set_text_color(202, 138, 4) # Yellow 600
        else:
            self.set_text_color(220, 38, 38) # Red 600
            
        self.cell(0, 15, f"FINAL DECISION: {decision}", ln=True, align="C")
        self.set_text_color(0, 0, 0)
        
        self.add_text(f"Confidence Level: {confidence:.0%}", style='B')
        self.add_text(f"Rationale: {decision_result['rationale']}")
        
        self.ln(5)
        
    def add_plot(self, fig, title="Visualization", width=170):
        self.add_section_title(title)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            try:
                pio.write_image(fig, tmp_file.name, format="png", width=1000, height=600)
                self.image(tmp_file.name, w=width)
                self.ln(10)
            except Exception as e:
                self.add_text(f"[Plot Generation Error: {e}]")
            finally:
                pass
        
        try:
            os.remove(tmp_file.name)
        except:
            pass

    def add_decision_memo(self, memo_text: str):
        self.add_page()
        self.add_section_title("Detailed Decision Memo")
        self.set_font("Courier", size=10)
        self.multi_cell(0, 5, memo_text)

    def save_report(self):
        return self.output(dest="S").encode("latin-1", errors="replace")

"""
Dataset Ethical Triage (DET) v3 - Visualizations

Creates comprehensive visualization suite for DET assessment results.
"""

from typing import Dict, List, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class DETVisualizations:
    """
    Generates premium visualizations for DET assessment results.
    """
    
    # Pro Dark Minimal Palette
    COLORS = {
        'PRIMARY': '#6366f1',    # Indigo
        'GREEN': '#4ade80',     # Mint/Green
        'YELLOW': '#facc15',    # Amber
        'RED': '#f87171',       # Rose/Red
        'BG': '#0f172a',        # Deep Slate
        'CARD': '#1e293b',
        'BORDER': '#334155',
        'TEXT': '#f8fafc',
        'TEXT_DIM': '#94a3b8'
    }
    
    @staticmethod
    def create_radar_chart(metric_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Create a clean radar chart for the Pro Dark theme.
        """
        metrics = []
        scores = []
        
        for metric_name, result in metric_results.items():
            score = result.get('score')
            if score is None: continue
            metrics.append(metric_name)
            if metric_name in ['k-Anon', 'FOI', 'SPA', 'DAI']:
                normalized = min(score / 20, 1.0) if metric_name == 'k-Anon' else score
            else:
                normalized = 1 - score if score <= 1 else 0
            scores.append(normalized)
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores, theta=metrics, fill='toself',
            name='Metric Profile',
            line_color=DETVisualizations.COLORS['PRIMARY'],
            fillcolor='rgba(99, 102, 241, 0.2)',
            marker=dict(size=6, color=DETVisualizations.COLORS['PRIMARY'])
        ))
        
        fig.update_layout(
            polar=dict(
                bgcolor=DETVisualizations.COLORS['BG'],
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=DETVisualizations.COLORS['BORDER'], tickfont=dict(color=DETVisualizations.COLORS['TEXT_DIM'])),
                angularaxis=dict(gridcolor=DETVisualizations.COLORS['BORDER'], tickfont=dict(color=DETVisualizations.COLORS['TEXT']))
            ),
            title=dict(text="DET Ethical Profile", font=dict(color=DETVisualizations.COLORS['TEXT'], size=16)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=60, b=40, l=40, r=40)
        )
        return fig
    
    @staticmethod
    def create_decision_matrix(metric_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Create a clean decision matrix heatmap for the Pro Dark theme.
        """
        metrics = list(metric_results.keys())
        flags = [metric_results[m].get('flag', 'YELLOW') for m in metrics]
        scores = [metric_results[m].get('score', 0) for m in metrics]
        
        colors = [1 if f == 'GREEN' else 0.5 if f == 'YELLOW' else 0 for f in flags]
        
        fig = go.Figure(data=go.Heatmap(
            z=[[c] for c in colors],
            y=metrics, x=['Status'],
            colorscale=[
                [0, 'rgba(239, 68, 68, 0.2)'],   # Red
                [0.5, 'rgba(234, 179, 8, 0.2)'], # Yellow
                [1, 'rgba(34, 197, 94, 0.2)']    # Green
            ],
            showscale=False,
            xgap=2, ygap=2,
            text=[[f] for f in flags]
        ))
        
        for i, (metric, flag) in enumerate(zip(metrics, flags)):
            txt_color = DETVisualizations.COLORS['RED'] if flag == 'RED' else \
                        DETVisualizations.COLORS['YELLOW'] if flag == 'YELLOW' else \
                        DETVisualizations.COLORS['GREEN']
            fig.add_annotation(
                x=0, y=i, text=f"<b>{flag}</b>",
                showarrow=False, font=dict(color=txt_color, size=11)
            )
        
        fig.update_layout(
            title=dict(text="Flag Matrix", font=dict(color=DETVisualizations.COLORS['TEXT'], size=16)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showticklabels=False),
            yaxis=dict(autorange='reversed', tickfont=dict(color=DETVisualizations.COLORS['TEXT_DIM'])),
            margin=dict(t=50, b=20, l=100, r=20)
        )
        return fig
    
    @staticmethod
    def create_category_summary(metric_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Create category summary bar chart.
        """
        categories = {
            'Core Strategy': ['URS', 'AOI', 'DMI', 'k-Anon', 'HRS'],
            'Advanced Analysis': ['FOI', 'FPC', 'CPA', 'SPA', 'DAI']
        }
        
        category_flags = {}
        for cat_name, cat_metrics in categories.items():
            f_list = [metric_results[m].get('flag', 'YELLOW') for m in cat_metrics if m in metric_results]
            category_flags[cat_name] = {'GREEN': f_list.count('GREEN'), 'YELLOW': f_list.count('YELLOW'), 'RED': f_list.count('RED')}
        
        fig = go.Figure()
        for f_type, color in [('GREEN', DETVisualizations.COLORS['GREEN']), 
                               ('YELLOW', DETVisualizations.COLORS['YELLOW']), 
                               ('RED', DETVisualizations.COLORS['RED'])]:
            fig.add_trace(go.Bar(
                name=f_type,
                x=list(categories.keys()),
                y=[category_flags[cat][f_type] for cat in categories.keys()],
                marker_color=color,
                width=0.4
            ))
        
        fig.update_layout(
            title=dict(text="Metric Distribution by Phase", font=dict(size=18, color=DETVisualizations.COLORS['TEXT'])),
            barmode='stack',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                gridcolor=DETVisualizations.COLORS['BORDER'], 
                title=dict(text="Metric Count", font=dict(color=DETVisualizations.COLORS['TEXT'])), 
                tickfont=dict(color=DETVisualizations.COLORS['TEXT_DIM'])
            ),
            xaxis=dict(tickfont=dict(color=DETVisualizations.COLORS['TEXT'])),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=DETVisualizations.COLORS['TEXT']))
        )
        
        return fig
    
    @staticmethod
    def create_decision_flow(decision_result: Dict[str, Any]) -> go.Figure:
        """
        Create decision flow diagram using Sankey.
        """
        decision = decision_result['decision']
        confidence = decision_result['confidence']
        flag_counts = decision_result['flag_counts']
        
        node_labels = ['Total Analysis', f"{flag_counts['GREEN']} GREEN", f"{flag_counts['YELLOW']} YELLOW", f"{flag_counts['RED']} RED", f"{decision}"]
        node_colors = [DETVisualizations.COLORS['TEXT'], DETVisualizations.COLORS['GREEN'], DETVisualizations.COLORS['YELLOW'], DETVisualizations.COLORS['RED'], DETVisualizations.COLORS['PRIMARY']]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=20, thickness=25, line=dict(color='white', width=1), label=node_labels, color=node_colors),
            link=dict(
                source=[0, 0, 0, 1, 2, 3],
                target=[1, 2, 3, 4, 4, 4],
                value=[flag_counts['GREEN'], flag_counts['YELLOW'], flag_counts['RED'], flag_counts['GREEN'], flag_counts['YELLOW'], flag_counts['RED']],
                color='rgba(150, 150, 150, 0.2)'
            )
        )])
        
        fig.update_layout(
            title=dict(text="Algorithmic Decision Pipeline", font=dict(size=18, color=DETVisualizations.COLORS['TEXT'])),
            font=dict(size=12, color=DETVisualizations.COLORS['TEXT']),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=60, b=40, l=40, r=40)
        )
        
        return fig

    @staticmethod
    def create_comprehensive_dashboard(metric_results: Dict[str, Dict[str, Any]],
                                      decision_result: Dict[str, Any]) -> go.Figure:
        """
        Dummy for now, as we use individual renders in the new UI.
        """
        return DETVisualizations.create_radar_chart(metric_results)

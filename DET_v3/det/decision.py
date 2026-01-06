"""DET v3 Decision Engine - Determines dataset bias level based on metric flags."""

from typing import Dict, List, Any, Tuple
from datetime import datetime


class DETDecisionEngine:
    """Rule-based decision engine for ethical dataset triage."""
    
    CRITICAL_METRICS = {'URS', 'AOI', 'HRS', 'k_anonymity'}
    
    def make_decision(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze metric flags and return bias decision with explanation."""
        flags = {name: result.get('flag', 'YELLOW') for name, result in metrics.items()}
        counts = self._count_flags(flags)
        
        decision, rationale = self._evaluate(flags, counts)
        actions = self._get_actions(flags, decision)
        memo = self._generate_memo(decision, rationale, metrics, flags, actions)
        
        return {
            'decision': decision,
            'rationale': rationale,
            'action_items': actions,
            'memo': memo,
            'flag_counts': counts
        }
    
    def _count_flags(self, flags: Dict[str, str]) -> Dict[str, int]:
        return {
            'red': sum(1 for f in flags.values() if f == 'RED'),
            'yellow': sum(1 for f in flags.values() if f == 'YELLOW'),
            'green': sum(1 for f in flags.values() if f == 'GREEN')
        }
    
    def _evaluate(self, flags: Dict[str, str], counts: Dict[str, int]) -> Tuple[str, str]:
        red, yellow, green = counts['red'], counts['yellow'], counts['green']
        red_list = [n for n, f in flags.items() if f == 'RED']
        yellow_list = [n for n, f in flags.items() if f == 'YELLOW']
        critical_red = [m for m in red_list if m in self.CRITICAL_METRICS]
        
        # SIGNIFICANT_BIAS: 3+ RED or 2+ RED with critical metric
        if red >= 3:
            return ('SIGNIFICANT_BIAS', f"Multiple critical issues ({red} RED: {', '.join(red_list)})")
        
        if red >= 2 and critical_red:
            return ('SIGNIFICANT_BIAS', f"Critical issues in {', '.join(critical_red)} with {red} RED flags")
        
        # MODERATE_BIAS: 1-2 RED or 4+ YELLOW
        if red == 2:
            return ('MODERATE_BIAS', f"Bias concerns in {', '.join(red_list)}")
        
        if red == 1:
            return ('MODERATE_BIAS', f"Bias concern in {red_list[0]}")
        
        if yellow >= 4:
            return ('MODERATE_BIAS', f"Multiple warnings ({yellow} YELLOW: {', '.join(yellow_list[:3])})")
        
        # NO_BIAS: No RED flags
        if green >= 8:
            return ('NO_BIAS', f"Dataset passes assessment ({green}/10 GREEN)")
        
        if green >= 6:
            return ('NO_BIAS', f"Dataset acceptable ({green} GREEN, {yellow} YELLOW)")
        
        if yellow <= 3:
            return ('NO_BIAS', f"Dataset acceptable with monitoring")
        
        return ('MODERATE_BIAS', f"Multiple areas need attention ({yellow} YELLOW)")
    
    def _get_actions(self, flags: Dict[str, str], decision: str) -> List[str]:
        base_actions = {
            'SIGNIFICANT_BIAS': "🔴 Significant bias - intervention required",
            'MODERATE_BIAS': "🟡 Moderate bias - consider mitigation",
            'NO_BIAS': "🟢 No significant bias detected"
        }
        
        metric_actions = {
            'URS': ("Apply oversampling or collect more data", "Monitor minority groups"),
            'AOI': ("Audit labels for bias", "Investigate outcome disparity"),
            'DMI': ("Use stratified imputation", "Run MCAR test"),
            'k_anonymity': ("Apply k-anonymization", "Generalize quasi-identifiers"),
            'HRS': ("Require ethics review", "Add fairness constraints"),
            'FOI': ("Consider subgroup models", "Monitor feature importance"),
            'FPC': ("Use fairness-aware learning", "Track per-group accuracy"),
            'CPA': ("Remove proxy features", "Monitor for disparate impact"),
            'SPA': ("Remove correlated features", "Consider adversarial debiasing"),
            'DAI': ("Apply stratified sampling", "Use sampling weights")
        }
        
        actions = [base_actions[decision]]
        for metric, flag in flags.items():
            if flag in ['RED', 'YELLOW'] and metric in metric_actions:
                idx = 0 if flag == 'RED' else 1
                actions.append(f"[{metric}] {metric_actions[metric][idx]}")
        
        return actions
    
    def _generate_memo(self, decision: str, rationale: str,
                       metrics: Dict, flags: Dict, actions: List[str]) -> str:
        badges = {'NO_BIAS': '🟢 NO BIAS', 'MODERATE_BIAS': '🟡 MODERATE BIAS', 
                  'SIGNIFICANT_BIAS': '🔴 SIGNIFICANT BIAS'}
        
        counts = self._count_flags(flags)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        
        metric_table = "\n".join(
            f"| {n} | {fmt(r.get('score', 'N/A'))} | {r.get('flag')} |"
            for n, r in metrics.items()
        )
        
        return f"""# DET v3 Decision Memo

**Generated:** {timestamp}

## Decision: {badges[decision]}
**Confidence:** {confidence:.0%}
**Rationale:** {rationale}

## Metrics
| Metric | Score | Flag |
|--------|-------|------|
{metric_table}

## Flag Summary
🟢 GREEN: {counts['green']} | 🟡 YELLOW: {counts['yellow']} | 🔴 RED: {counts['red']}

## Actions
{chr(10).join(f"- {a}" for a in actions)}

---
*Generated by DET v3*
"""


def make_decision(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience function for making decisions."""
    return DETDecisionEngine().make_decision(metrics)

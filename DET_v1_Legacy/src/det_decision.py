"""
Dataset Ethical Triage (DET) v3 - Decision Engine

Implements branching decision logic for PROCEED/MITIGATE/REJECT based on
10 metric flags (GREEN/YELLOW/RED).

Decision Rules (from DET v3 spec):
- red >= 2 → REJECT
- red = 1 & HRS=RED → REJECT (harm override)
- red >= 1 OR yellow >= 3 → MITIGATE + actions
- else → PROCEED
"""

from typing import Dict, List, Any, Tuple
from datetime import datetime


class DETDecisionEngine:
    """
    Makes triage decisions based on metric flags and generates decision memos.
    """
    
    # Default thresholds for all 10 metrics
    DEFAULT_THRESHOLDS = {
        # Core metrics
        'URS': {'yellow': 0.20, 'red': 0.10},
        'AOI': {'yellow': 0.15, 'red': 0.25},
        'DMI': {'yellow': 0.10, 'red': 0.20},
        'k-Anon': {'yellow': 10, 'red': 5},  # Inverted (higher is better)
        'HRS': {'yellow': 0.25, 'red': 0.50},
        
        # Advanced metrics
        'FOI': {'yellow': 0.85, 'red': 0.70},  # Inverted (higher is better)
        'FPC': {'yellow': 0.70, 'red': 0.50},
        'CPA': {'yellow': 0.30, 'red': 0.50},
        'SPA': {'yellow': 0.30, 'red': 0.15},  # Inverted (higher is better)
        'DAI': {'yellow': 0.85, 'red': 0.70},  # Inverted (higher is better)
    }
    
    # Mitigation actions mapped to metrics
    MITIGATION_ACTIONS = {
        'URS': [
            "Oversample underrepresented groups using SMOTE or similar techniques",
            "Collect additional data for minority classes/groups",
            "Consider stratified sampling in future data collection"
        ],
        'AOI': [
            "Investigate root causes of outcome disparity (confounding vs. bias)",
            "Apply fairness constraints during model training (e.g., demographic parity)",
            "Conduct subgroup analysis to understand disparities",
            "Consider separate models or calibration per group"
        ],
        'DMI': [
            "Use stratified imputation (impute within groups, not globally)",
            "Investigate why missingness differs by group (MNAR analysis)",
            "Consider multiple imputation with group-specific models",
            "Document missingness patterns in model card"
        ],
        'k-Anon': [
            "Apply generalization to quasi-identifiers (e.g., age → age groups)",
            "Suppress rare combinations in quasi-ID space",
            "Apply differential privacy (ε-DP) with appropriate privacy budget",
            "Remove or mask high-risk quasi-identifiers"
        ],
        'HRS': [
            "Implement fairness-aware training (equalized odds, equal opportunity)",
            "Add human-in-the-loop review for high-risk predictions",
            "Deploy with continuous fairness monitoring",
            "Consider rejecting dataset if harm cannot be mitigated"
        ],
        'FOI': [
            "Remove sensitive attribute from training data",
            "Apply adversarial debiasing to reduce dependence",
            "Use causal inference to distinguish bias from legitimate correlation"
        ],
        'FPC': [
            "Balance fairness and performance trade-offs explicitly",
            "Use multi-objective optimization during training",
            "Document performance disparities across groups"
        ],
        'CPA': [
            "Remove or mask proxy features with high MI scores",
            "Apply feature transformation to reduce proxy leakage",
            "Use fairness-aware feature selection"
        ],
        'SPA': [
            "Remove features that enable sensitive attribute prediction",
            "Apply representation learning to obfuscate sensitive information",
            "Use privacy-preserving feature engineering"
        ],
        'DAI': [
            "Reweight samples to match target distribution",
            "Apply importance sampling during training",
            "Collect additional data to improve distributional alignment"
        ]
    }
    
    def __init__(self, thresholds: Dict[str, Dict[str, float]] = None):
        """
        Initialize decision engine.
        
        Args:
            thresholds: Optional custom thresholds (defaults to DEFAULT_THRESHOLDS)
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
    
    def make_decision(self, metric_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make triage decision based on metric flags.
        
        Args:
            metric_results: Dict mapping metric names to their result dicts
                           (from DETMetricsCalculator.calculate_all_metrics())
        
        Returns:
            Dict with decision, confidence, action items, and memo
        """
        # Extract flags
        flags = {name: result['flag'] for name, result in metric_results.items() 
                if result.get('flag')}
        
        # Count flags
        red_count = sum(1 for flag in flags.values() if flag == 'RED')
        yellow_count = sum(1 for flag in flags.values() if flag == 'YELLOW')
        green_count = sum(1 for flag in flags.values() if flag == 'GREEN')
        
        # Apply decision rules
        decision, confidence, rationale = self._apply_decision_rules(
            flags, red_count, yellow_count, green_count
        )
        
        # Generate action items
        action_items = self._generate_action_items(flags, decision)
        
        # Generate decision memo
        memo = self._generate_decision_memo(
            decision, confidence, rationale, metric_results, flags, action_items
        )
        
        return {
            'decision': decision,
            'confidence': confidence,
            'rationale': rationale,
            'flags': flags,
            'flag_counts': {
                'RED': red_count,
                'YELLOW': yellow_count,
                'GREEN': green_count
            },
            'action_items': action_items,
            'memo': memo,
            'timestamp': datetime.now().isoformat()
        }
    
    def _apply_decision_rules(self, flags: Dict[str, str], 
                             red_count: int, yellow_count: int, 
                             green_count: int) -> Tuple[str, float, str]:
        """
        Apply DET v3 decision rules.
        
        Returns:
            Tuple of (decision, confidence, rationale)
        """
        # Rule 1: red >= 2 → REJECT
        if red_count >= 2:
            confidence = 0.9 if red_count >= 3 else 0.8
            rationale = f"{red_count} metrics flagged RED (≥2 threshold). Multiple severe issues detected."
            return 'REJECT', confidence, rationale
        
        # Rule 2: red = 1 & HRS=RED → REJECT (harm override)
        if red_count == 1 and flags.get('HRS') == 'RED':
            confidence = 0.95
            rationale = "HRS flagged RED. High harm risk overrides other factors (harm override rule)."
            return 'REJECT', confidence, rationale
        
        # Rule 3: red >= 1 OR yellow >= 3 → MITIGATE
        if red_count >= 1 or yellow_count >= 3:
            if red_count == 1:
                confidence = 0.7
                rationale = f"1 RED flag ({self._get_red_metrics(flags)[0]}). Mitigation required before proceeding."
            else:
                confidence = 0.75
                rationale = f"{yellow_count} YELLOW flags. Multiple moderate concerns require mitigation."
            return 'MITIGATE', confidence, rationale
        
        # Rule 4: else → PROCEED
        if yellow_count <= 2:
            confidence = 0.9 if yellow_count == 0 else 0.8
            rationale = f"All metrics GREEN or minor YELLOW flags ({yellow_count}). Dataset suitable for training with standard monitoring."
        else:
            confidence = 0.7
            rationale = f"{yellow_count} YELLOW flags. Proceed with enhanced monitoring."
        
        return 'PROCEED', confidence, rationale
    
    def _get_red_metrics(self, flags: Dict[str, str]) -> List[str]:
        """Get list of metrics flagged RED."""
        return [name for name, flag in flags.items() if flag == 'RED']
    
    def _get_yellow_metrics(self, flags: Dict[str, str]) -> List[str]:
        """Get list of metrics flagged YELLOW."""
        return [name for name, flag in flags.items() if flag == 'YELLOW']
    
    def _generate_action_items(self, flags: Dict[str, str], decision: str) -> List[str]:
        """
        Generate specific action items based on flags and decision.
        
        Args:
            flags: Dict mapping metric names to flags
            decision: PROCEED/MITIGATE/REJECT
        
        Returns:
            List of actionable recommendations
        """
        actions = []
        
        if decision == 'REJECT':
            actions.append("🔴 **REJECT DATASET** - Do not proceed with model training")
            actions.append("Conduct root cause analysis for all RED-flagged metrics")
            actions.append("Consider dataset redesign or alternative data sources")
        
        # Add metric-specific actions for RED flags
        red_metrics = self._get_red_metrics(flags)
        for metric in red_metrics:
            if metric in self.MITIGATION_ACTIONS:
                actions.extend([f"  - {action}" for action in self.MITIGATION_ACTIONS[metric]])
        
        # Add metric-specific actions for YELLOW flags (if MITIGATE)
        if decision == 'MITIGATE':
            yellow_metrics = self._get_yellow_metrics(flags)
            for metric in yellow_metrics:
                if metric in self.MITIGATION_ACTIONS:
                    # Add first action only for YELLOW (less critical)
                    actions.append(f"  - {self.MITIGATION_ACTIONS[metric][0]}")
        
        # Add general monitoring for PROCEED
        if decision == 'PROCEED':
            actions.append("✅ **PROCEED** with standard fairness monitoring")
            actions.append("Implement continuous fairness audits post-deployment")
            actions.append("Re-evaluate DET metrics quarterly or after data updates")
        
        return actions
    
    def _generate_decision_memo(self, decision: str, confidence: float, 
                                rationale: str, metric_results: Dict[str, Dict[str, Any]],
                                flags: Dict[str, str], action_items: List[str]) -> str:
        """
        Generate structured decision memo.
        
        Returns:
            Formatted markdown memo
        """
        memo_lines = [
            "# DATASET ETHICAL TRIAGE (DET) - DECISION MEMO",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Decision**: **{decision}**",
            f"**Confidence**: {confidence:.0%}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            rationale,
            "",
            "---",
            "",
            "## Metric Summary",
            "",
            "| Metric | Score | Flag | Interpretation |",
            "|--------|-------|------|----------------|"
        ]
        
        # Add metric rows
        for metric_name, result in metric_results.items():
            score = result.get('score', 'N/A')
            flag = result.get('flag', 'N/A')
            interpretation = result.get('interpretation', '')
            
            # Format score
            if isinstance(score, float):
                score_str = f"{score:.3f}"
            else:
                score_str = str(score)
            
            # Emoji for flag
            flag_emoji = {
                'GREEN': '🟢',
                'YELLOW': '🟡',
                'RED': '🔴'
            }.get(flag, '⚪')
            
            # Truncate interpretation for table
            interp_short = interpretation.split(':')[0] if ':' in interpretation else interpretation
            interp_short = interp_short[:50] + '...' if len(interp_short) > 50 else interp_short
            
            memo_lines.append(f"| {metric_name} | {score_str} | {flag_emoji} {flag} | {interp_short} |")
        
        memo_lines.extend([
            "",
            "---",
            "",
            "## Flag Distribution",
            "",
            f"- 🔴 **RED**: {sum(1 for f in flags.values() if f == 'RED')} metrics",
            f"- 🟡 **YELLOW**: {sum(1 for f in flags.values() if f == 'YELLOW')} metrics",
            f"- 🟢 **GREEN**: {sum(1 for f in flags.values() if f == 'GREEN')} metrics",
            "",
            "---",
            "",
            "## Recommended Actions",
            ""
        ])
        
        # Add action items
        for action in action_items:
            memo_lines.append(action)
        
        memo_lines.extend([
            "",
            "---",
            "",
            "## Decision Justification",
            "",
            "**Decision Rules Applied** (DET v3):",
            "1. `red >= 2` → REJECT",
            "2. `red = 1 & HRS=RED` → REJECT (harm override)",
            "3. `red >= 1 OR yellow >= 3` → MITIGATE + actions",
            "4. `else` → PROCEED",
            "",
            f"**Outcome**: {decision} (Rule {self._get_applied_rule(flags)})",
            "",
            "---",
            "",
            "## Limitations & Epistemic Humility",
            "",
            "- Thresholds are calibrated defaults; domain-specific calibration recommended",
            "- Metrics detect signals, not root causes; further investigation required",
            "- Decision framework is one tool among many; stakeholder engagement essential",
            "- Fairness is context-dependent; local priorities may differ",
            "",
            "---",
            "",
            "## Next Steps",
            "",
            "1. Review this memo with data ethics board and domain experts",
            "2. Implement recommended mitigations (if MITIGATE)",
            "3. Re-run DET assessment after mitigation",
            "4. Document decision and rationale in model card",
            "5. Establish continuous monitoring plan (if PROCEED)",
            "",
            "---",
            "",
            "*Generated by DET v3 Framework*"
        ])
        
        return '\n'.join(memo_lines)
    
    def _get_applied_rule(self, flags: Dict[str, str]) -> int:
        """Determine which decision rule was applied."""
        red_count = sum(1 for f in flags.values() if f == 'RED')
        yellow_count = sum(1 for f in flags.values() if f == 'YELLOW')
        
        if red_count >= 2:
            return 1
        elif red_count == 1 and flags.get('HRS') == 'RED':
            return 2
        elif red_count >= 1 or yellow_count >= 3:
            return 3
        else:
            return 4

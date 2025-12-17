# -*- coding: utf-8 -*-
"""
Finance Domain Graders

This package provides specialized graders for evaluating financial analysis
and research content. All graders follow the new RM-Gallery framework architecture.

Modules:
    - event_interpretation: Event identification and analysis graders
    - industry_research: Industry characteristics, risk, and comparison graders
    - macro_analysis: Macroeconomic analysis and concept explanation graders
"""

from tutorials.finance.event_interpretation import (
    EventAnalysisGrader,
    EventIdentificationGrader,
)
from tutorials.finance.industry_research import (
    CharacteristicsAnalysisGrader,
    RiskAnalysisGrader,
    UnderlyingComparisonGrader,
)
from tutorials.finance.macro_analysis import (
    ConceptExplanationGrader,
    MacroAnalysisGrader,
)

__all__ = [
    # Event Interpretation
    "EventAnalysisGrader",
    "EventIdentificationGrader",
    # Industry Research
    "CharacteristicsAnalysisGrader",
    "RiskAnalysisGrader",
    "UnderlyingComparisonGrader",
    # Macro Analysis
    "ConceptExplanationGrader",
    "MacroAnalysisGrader",
]

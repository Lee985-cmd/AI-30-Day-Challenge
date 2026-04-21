"""
智能投研助手 - 多智能体协作系统
基于 LangChain + CrewAI 实现自动化投资研究
"""

from .researcher import ResearcherAgent
from .analyst import AnalystAgent
from .writer import WriterAgent
from .risk_manager import RiskManagerAgent
from .orchestrator import InvestmentResearchOrchestrator

__version__ = "1.0.0"
__author__ = "Lee"

__all__ = [
    "ResearcherAgent",
    "AnalystAgent", 
    "WriterAgent",
    "RiskManagerAgent",
    "InvestmentResearchOrchestrator"
]

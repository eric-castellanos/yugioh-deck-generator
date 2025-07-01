"""
WindBot Automation Package

This package provides complete automation for YGOPro duel simulation using WindBot.
It enables .ydk vs .ydk automated duels for ML/AI research and deck evaluation.

Main Components:
- windbot_wrapper: Python wrapper for WindBot C# application
- ygopro_server: YGOPro server management and monitoring
- automation_integration: Complete automation system combining server and bots

Usage:
    from windbot_automation import YGOProAutomation, AutomationConfig
    
    config = AutomationConfig()
    automation = YGOProAutomation(config)
    
    result = automation.simulate_single_duel(
        deck1_path="deck1.ydk",
        deck2_path="deck2.ydk"
    )
"""

from .windbot_wrapper import WindBotWrapper, DuelConfig, DuelResult
from .ygopro_server import YGOProServer, ServerConfig
from .automation_integration import YGOProAutomation, AutomationConfig

__all__ = [
    'WindBotWrapper', 'DuelConfig', 'DuelResult',
    'YGOProServer', 'ServerConfig', 
    'YGOProAutomation', 'AutomationConfig'
]

__version__ = "1.0.0"
__author__ = "YuGiOh Deck Generator Project"

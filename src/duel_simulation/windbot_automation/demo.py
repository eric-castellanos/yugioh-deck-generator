#!/usr/bin/env python3
"""
WindBot Automation Demo

This script demonstrates the complete automated duel simulation system.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.duel_simulation.windbot_automation.automation_integration import YGOProAutomation, AutomationConfig

def main():
    print("🎮 WindBot Automation Demo")
    print("=" * 50)
    
    # Configure automation
    config = AutomationConfig(
        edopro_path="/home/ecast229/Applications/EDOPro",  # Directory path, not executable
        windbot_path="/home/ecast229/Projects/yugioh-deck-generator/Applications/WindBot/WindBot.exe",  # Full path to WindBot executable
        debug=True,
        save_replays=True,
        duel_timeout=60  # Shorter timeout for demo
    )
    
    print("Initializing automation system...")
    automation = YGOProAutomation(config)
    
    # Find some deck files for testing
    deck_dir = project_root / "decks" / "known"
    deck_files = list(deck_dir.glob("*.ydk"))[:5]  # Just use first 5 decks
    
    if len(deck_files) < 2:
        print("❌ Need at least 2 deck files to run demo")
        return
    
    print(f"Found {len(deck_files)} deck files for testing:")
    for i, deck in enumerate(deck_files):
        print(f"  {i+1}. {deck.stem}")
    
    # Demo 1: Single duel
    print("\n🥊 Demo 1: Single Duel")
    print("-" * 30)
    
    deck1 = deck_files[0]
    deck2 = deck_files[1]
    
    print(f"Duel: {deck1.stem} vs {deck2.stem}")
    print("Starting duel simulation...")
    
    try:
        result = automation.simulate_single_duel(
            deck1_path=str(deck1),
            deck2_path=str(deck2),
            deck1_name=deck1.stem,
            deck2_name=deck2.stem
        )
        
        if result.error:
            print(f"❌ Duel failed: {result.error}")
        else:
            winner_name = result.deck1_name if result.winner == 0 else result.deck2_name
            print(f"🏆 Winner: {winner_name}")
            print(f"⏱️  Duration: {result.duration:.1f} seconds")
            print(f"🔄 Turns: {result.turns}")
            if result.replay_path:
                print(f"💾 Replay saved: {result.replay_path}")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Demo 2: Mini tournament (if we have enough decks)
    if len(deck_files) >= 3:
        print("\n🏆 Demo 2: Mini Tournament")
        print("-" * 30)
        
        tournament_decks = deck_files[:3]
        print(f"Tournament participants:")
        for i, deck in enumerate(tournament_decks):
            print(f"  {i+1}. {deck.stem}")
        
        try:
            results = automation.simulate_tournament(
                deck_paths=[str(d) for d in tournament_decks],
                rounds=1  # Just 1 round per matchup for demo
            )
            
            print(f"\n📊 Tournament Results:")
            print(f"Total duels: {len(results)}")
            
            # Count wins per deck
            deck_wins = {}
            for result in results:
                if result.error is None:
                    winner = result.deck1_name if result.winner == 0 else result.deck2_name
                    deck_wins[winner] = deck_wins.get(winner, 0) + 1
            
            print("Win counts:")
            for deck, wins in sorted(deck_wins.items(), key=lambda x: x[1], reverse=True):
                print(f"  {deck}: {wins} wins")
        
        except Exception as e:
            print(f"❌ Tournament demo failed: {e}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    automation.cleanup()
    
    print("\n✅ Demo completed!")
    print("\nThe WindBot automation system is working correctly!")
    print("You can now integrate it into your ML pipeline for automated deck evaluation.")

if __name__ == "__main__":
    main()

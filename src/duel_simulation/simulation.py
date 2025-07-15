import subprocess
import time
import os
from pathlib import Path
import platform

# Constants
DECK1 = "AI_SnakeEyes.ydk"
DECK2 = "AI_SnakeEyes.ydk"
YGOPRO_PATH = "/bin/ygopro"  # inside edopro container
PORT = 7911

# Helper to run WindBot
def run_windbot(name, deck_path, hand, is_first, port, is_mcts=False):
    return subprocess.Popen([
        "docker", "exec", "windbot", "mono", "/windbot/WindBot.exe",
        f"Deck=AIBase",
        f"DeckFile={deck_path}",
        f"ExportPath=/windbot/GameData",
        f"Name={name}",
        f"Hand={hand}",
        f"IsTraining=True",
        f"ShouldUpdate=True",
        f"ShouldRecord=True",
        f"TotalGames=1",
        f"IsFirst={str(is_first)}",
        f"Id={name}",
        f"Port={port}",
        f"IsMCTS={str(is_mcts)}",
        f"FixedRNG=False"
    ], stdout=subprocess.DEVNULL)

def main():
    print("🚀 Launching test duel simulation...")

    # Step 1: Clean up old GameData (optional)
    subprocess.run(["rm", "-rf", "./GameData/*"], shell=True)

    # Step 2: Start ygopro (EDOPro)
    print("🧠 Starting EDOPro engine...")
    edopro = subprocess.Popen([
        "docker", "exec", "edopro", YGOPRO_PATH, "-p", str(PORT)
    ], stdout=subprocess.DEVNULL)

    time.sleep(5)  # Wait for server to start

    # Step 3: Run WindBot AIs
    print("🤖 Starting WindBot A and B...")
    bot1 = run_windbot("Bot1", f"/windbot/decks/{DECK1}", hand=2, is_first=True, port=PORT, is_mcts=True)
    bot2 = run_windbot("Bot2", f"/windbot/decks/{DECK2}", hand=3, is_first=False, port=PORT, is_mcts=False)

    # Step 4: Wait for both bots to finish
    bot1.wait()
    bot2.wait()

    print("✅ Duel complete. Killing EDOPro...")
    edopro.terminate()

    print("📂 GameData should now contain results.")

if __name__ == "__main__":
    main()

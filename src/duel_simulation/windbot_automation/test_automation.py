#!/usr/bin/env python3
"""
Test script for WindBot automation system.

This script validates the installation and basic functionality
of the WindBot automation components.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from duel_simulation.windbot_automation.windbot_wrapper import WindBotWrapper, DuelConfig, DuelResult
        print("✓ WindBot wrapper imports OK")
    except ImportError as e:
        print(f"✗ WindBot wrapper import failed: {e}")
        return False
    
    try:
        from duel_simulation.windbot_automation.ygopro_server import YGOProServer, ServerConfig
        print("✓ YGOPro server imports OK")
    except ImportError as e:
        print(f"✗ YGOPro server import failed: {e}")
        return False
    
    try:
        from duel_simulation.windbot_automation.automation_integration import YGOProAutomation, AutomationConfig
        print("✓ Automation integration imports OK")
    except ImportError as e:
        print(f"✗ Automation integration import failed: {e}")
        return False
    
    return True

def test_edopro_path():
    """Test that EDOPro installation is found."""
    print("\nTesting EDOPro installation...")
    
    edopro_path = Path("/home/ecast229/Applications/EDOPro")
    
    if not edopro_path.exists():
        print(f"✗ EDOPro not found at {edopro_path}")
        return False
    
    print(f"✓ EDOPro directory found at {edopro_path}")
    
    # Look for executable
    exe_candidates = [
        edopro_path / "EDOPro",
        edopro_path / "ygopro",
        edopro_path / "ygopro.exe"
    ]
    
    for exe in exe_candidates:
        if exe.exists():
            print(f"✓ EDOPro executable found: {exe}")
            return True
    
    print("✗ No EDOPro executable found")
    return False

def test_mono():
    """Test that Mono is installed and working."""
    print("\nTesting Mono installation...")
    
    import subprocess
    
    try:
        result = subprocess.run(["mono", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ Mono installed: {version_line}")
            return True
        else:
            print(f"✗ Mono version check failed: {result.stderr}")
            return False
    
    except FileNotFoundError:
        print("✗ Mono not found. Install with: sudo apt-get install mono-complete")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Mono version check timed out")
        return False

def test_windbot_setup():
    """Test WindBot setup."""
    print("\nTesting WindBot setup...")
    
    # Check for setup script
    setup_script = Path(__file__).parent / "setup_windbot.sh"
    if not setup_script.exists():
        print(f"✗ Setup script not found: {setup_script}")
        return False
    
    print(f"✓ Setup script found: {setup_script}")
    
    # Check if WindBot is already built
    project_root = Path(__file__).parent.parent.parent.parent
    windbot_path = project_root / "Applications" / "WindBot" / "WindBot.exe"
    
    if windbot_path.exists():
        print(f"✓ WindBot already built: {windbot_path}")
        return True
    else:
        print(f"○ WindBot not built yet: {windbot_path}")
        print("  Run ./setup_windbot.sh to build WindBot")
        return False

def test_deck_files():
    """Test that sample deck files exist."""
    print("\nTesting deck files...")
    
    project_root = Path(__file__).parent.parent.parent.parent
    deck_dir = project_root / "decks" / "known"
    
    if not deck_dir.exists():
        print(f"✗ Deck directory not found: {deck_dir}")
        return False
    
    print(f"✓ Deck directory found: {deck_dir}")
    
    # Look for .ydk files
    ydk_files = list(deck_dir.glob("*.ydk"))
    
    if len(ydk_files) == 0:
        print("✗ No .ydk files found")
        return False
    
    print(f"✓ Found {len(ydk_files)} deck files")
    
    # Show a few examples
    for i, ydk_file in enumerate(ydk_files[:3]):
        print(f"  - {ydk_file.name}")
    
    if len(ydk_files) > 3:
        print(f"  ... and {len(ydk_files) - 3} more")
    
    return True

def test_basic_functionality():
    """Test basic functionality of the automation system."""
    print("\nTesting basic functionality...")
    
    try:
        from duel_simulation.windbot_automation.windbot_wrapper import DuelConfig, DuelResult
        
        # Test DuelConfig creation
        config = DuelConfig(
            deck1_path="/test/deck1.ydk",
            deck2_path="/test/deck2.ydk",
            deck1_name="TestDeck1",
            deck2_name="TestDeck2"
        )
        print("✓ DuelConfig creation works")
        
        # Test DuelResult creation
        result = DuelResult(
            winner=0,
            turns=10,
            duration=120.5,
            deck1_name="TestDeck1",
            deck2_name="TestDeck2"
        )
        print("✓ DuelResult creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_server_component():
    """Test YGOPro server component."""
    print("\nTesting YGOPro server component...")
    
    try:
        from duel_simulation.windbot_automation.ygopro_server import YGOProServer, ServerConfig
        
        # Test server config
        config = ServerConfig(port=7911, host="127.0.0.1")
        print("✓ ServerConfig creation works")
        
        # Test server initialization (without actually starting)
        try:
            server = YGOProServer()
            print("✓ YGOProServer initialization works")
            return True
        except Exception as e:
            print(f"○ YGOProServer initialization issue: {e}")
            print("  This may be normal if EDOPro is not fully configured")
            return False
        
    except Exception as e:
        print(f"✗ Server component test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("WindBot Automation System Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("EDOPro Installation", test_edopro_path),
        ("Mono Runtime", test_mono),
        ("WindBot Setup", test_windbot_setup),
        ("Deck Files", test_deck_files),
        ("Basic Functionality", test_basic_functionality),
        ("Server Component", test_server_component),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✓" if result else "✗"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! The automation system is ready to use.")
        print("\nNext steps:")
        print("1. Run ./setup_windbot.sh if WindBot is not built yet")
        print("2. Try the example in automation_integration.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
        print("\nCommon fixes:")
        print("- Install Mono: sudo apt-get install mono-complete")
        print("- Run WindBot setup: ./setup_windbot.sh")
        print("- Check EDOPro installation path")
        return 1

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.WARNING)
    
    sys.exit(main())

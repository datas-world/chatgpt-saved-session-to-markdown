#!/usr/bin/env python3
# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later

"""Simple pytest-compatible test runner for environments without pytest."""

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path


def discover_and_run_tests(test_file_path: Path) -> tuple[int, int]:
    """Discover and run all test functions in a test file.
    
    Args:
        test_file_path: Path to the test file
        
    Returns:
        Tuple of (passed_count, failed_count)
    """
    # Load the test module
    spec = importlib.util.spec_from_file_location("test_module", test_file_path)
    if spec is None or spec.loader is None:
        print(f"❌ Could not load test module: {test_file_path}")
        return 0, 1
        
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    # Find all test functions
    test_functions = []
    for name, obj in inspect.getmembers(test_module):
        if (inspect.isfunction(obj) and 
            name.startswith('test_') and 
            getattr(obj, '__module__', None) == test_module.__name__):
            test_functions.append((name, obj))
    
    if not test_functions:
        print(f"⚠️  No test functions found in {test_file_path}")
        return 0, 0
    
    print(f"📋 Discovered {len(test_functions)} test(s) in {test_file_path.name}")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, test_func in test_functions:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 60)
        
        try:
            test_func()
            print(f"✅ PASSED: {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {e}")
            print("   Traceback:")
            traceback.print_exc(limit=3)
            failed += 1
    
    return passed, failed


def main():
    """Main test runner entry point."""
    if len(sys.argv) != 2:
        print("Usage: python pytest_runner.py <test_file.py>")
        sys.exit(1)
    
    test_file = Path(sys.argv[1])
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)
    
    print("🚀 Simple pytest-compatible test runner")
    print(f"📁 Test file: {test_file}")
    
    passed, failed = discover_and_run_tests(test_file)
    
    print("\n" + "=" * 80)
    print(f"📊 TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed > 0:
        print("❌ Some tests failed")
        sys.exit(1)
    else:
        print("✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
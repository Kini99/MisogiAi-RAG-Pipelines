#!/usr/bin/env python3
"""
Simple test to verify the parameter fix works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_utils_import():
    """Test that data_utils can be imported and the function works."""
    try:
        from data_utils import load_training_data, create_sample_data
        
        print("✅ data_utils imported successfully")
        
        # Test the sample data creation
        texts, labels = create_sample_data()
        print(f"✅ Sample data created: {len(texts)} texts, {len(labels)} labels")
        print(f"Categories: {set(labels)}")
        
        # Test the load_training_data function with the fixed parameter
        texts_sample, labels_sample = load_training_data(use_sample=True)
        print(f"✅ load_training_data with use_sample=True: {len(texts_sample)} texts")
        
        # Test that the old parameter name would fail
        try:
            # This should fail if the fix is working correctly
            texts_old, labels_old = load_training_data(use_sample_data=True)
            print("❌ Old parameter name still works - this shouldn't happen!")
            return False
        except TypeError as e:
            print(f"✅ Old parameter name correctly fails: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ data_utils test failed: {e}")
        return False

def test_app_import():
    """Test that app.py can be imported without the parameter error."""
    try:
        # Import the specific function that was causing the error
        import app
        
        print("✅ app.py imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ app.py import failed: {e}")
        return False

def main():
    """Run the simple tests."""
    print("🔍 Testing the parameter fix...\n")
    
    tests = [
        ("Data Utils Import", test_data_utils_import),
        ("App Import", test_app_import),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    print("=" * 40)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! The parameter fix is working correctly.")
        print("\nThe main issue was:")
        print("- Function load_training_data() expected parameter 'use_sample'")
        print("- Code was calling it with 'use_sample_data'")
        print("- Fixed by changing all calls to use the correct parameter name")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 
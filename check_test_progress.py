import os
import time

print("Checking test progress...")
print("=" * 80)

# Check if output files exist
if os.path.exists("owlvit_test_output.log"):
    print("\nLast 20 lines of output:")
    print("-" * 80)
    with open("owlvit_test_output.log", "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        for line in lines[-20:]:
            print(line.rstrip())
else:
    print("\nNo output file found yet...")

if os.path.exists("owlvit_test_error.log"):
    with open("owlvit_test_error.log", "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        if content.strip():
            print("\nErrors:")
            print("-" * 80)
            print(content)

if os.path.exists("owlvit_test_results.json"):
    print("\n" + "=" * 80)
    print("TEST COMPLETED! Results saved to owlvit_test_results.json")
    print("=" * 80)
    import json
    with open("owlvit_test_results.json", "r") as f:
        results = json.load(f)
    print(f"\nTest Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"Test Precision: {results['test_precision']*100:.2f}%")
    print(f"Test Recall: {results['test_recall']*100:.2f}%")
    print(f"Test F1 Score: {results['test_f1']*100:.2f}%")
else:
    print("\nTest is still running... Results will be saved to 'owlvit_test_results.json'")

#!/usr/bin/env python3
"""
Verify Checkpoint Structure for MIMO Late Fusion Training
=========================================================

This script verifies that the checkpoint structure is properly configured
and shows the directories that will be used for saving training results.
"""

import os
from datetime import datetime

def verify_checkpoint_structure():
    """Verify and display the checkpoint directory structure"""
    
    print("="*70)
    print("MIMO LATE FUSION CHECKPOINT STRUCTURE VERIFICATION")
    print("="*70)
    
    # Main checkpoint directory
    main_checkpoint_dir = "checkpoint_4_mimamo"
    
    # Results directory structure
    results_dir = "using_mimo_late_result"
    subdirs = ["checkpoints", "plots", "logs", "optuna_studies"]
    
    print(f"\n📂 Main Checkpoint Directory:")
    print(f"   {main_checkpoint_dir}/")
    if os.path.exists(main_checkpoint_dir):
        print(f"   ✅ EXISTS")
        files = os.listdir(main_checkpoint_dir)
        if files:
            print(f"   📁 Contains {len(files)} files:")
            for file in files[:5]:  # Show first 5 files
                print(f"      - {file}")
            if len(files) > 5:
                print(f"      ... and {len(files) - 5} more files")
        else:
            print(f"   📂 Empty (ready for new checkpoints)")
    else:
        print(f"   ❌ DOES NOT EXIST")
        print(f"   🔧 Creating directory...")
        os.makedirs(main_checkpoint_dir, exist_ok=True)
        print(f"   ✅ Created successfully")
    
    print(f"\n📂 Results Directory Structure:")
    print(f"   {results_dir}/")
    
    for subdir in subdirs:
        full_path = os.path.join(results_dir, subdir)
        print(f"   ├── {subdir}/")
        
        if os.path.exists(full_path):
            print(f"   │   ✅ EXISTS")
            files = os.listdir(full_path)
            if files:
                print(f"   │   📁 Contains {len(files)} files")
                for file in files[:3]:  # Show first 3 files
                    print(f"   │      - {file}")
                if len(files) > 3:
                    print(f"   │      ... and {len(files) - 3} more files")
            else:
                print(f"   │   📂 Empty")
        else:
            print(f"   │   ❌ DOES NOT EXIST")
            print(f"   │   🔧 Creating...")
            os.makedirs(full_path, exist_ok=True)
            print(f"   │   ✅ Created")
    
    print(f"\n📋 Checkpoint File Naming Convention:")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"   Main Checkpoints:")
    print(f"   └── checkpoint_4_mimamo/best_combined_fusion_model_{timestamp}_acc0.XXXX.pth")
    print(f"   └── checkpoint_4_mimamo/best_optimized_fusion_model_{timestamp}_acc0.XXXX.pth")
    print(f"   ")
    print(f"   Backup Checkpoints:")
    print(f"   └── using_mimo_late_result/checkpoints/best_combined_fusion_model_{timestamp}_acc0.XXXX.pth")
    print(f"   └── using_mimo_late_result/checkpoints/best_optimized_fusion_model_{timestamp}_acc0.XXXX.pth")
    
    print(f"\n🎯 Key Features:")
    print(f"   • Dual checkpoint saving (main + backup)")
    print(f"   • Automatic hyperparameter optimization with Optuna")
    print(f"   • Comprehensive training visualization and logging")
    print(f"   • Organized results structure with documentation")
    
    print(f"\n📊 Training Progress will be saved to:")
    print(f"   • 📈 Plots: {results_dir}/plots/")
    print(f"   • 📝 Logs: {results_dir}/logs/") 
    print(f"   • 🔍 Optuna Studies: {results_dir}/optuna_studies/")
    print(f"   • 💾 Main Models: {main_checkpoint_dir}/")
    print(f"   • 🔄 Backup Models: {results_dir}/checkpoints/")
    
    print("\n" + "="*70)
    print("✅ CHECKPOINT STRUCTURE VERIFICATION COMPLETE")
    print("="*70)
    
    return {
        'main_checkpoint_dir': main_checkpoint_dir,
        'results_dir': results_dir,
        'all_ready': os.path.exists(main_checkpoint_dir) and os.path.exists(results_dir)
    }

if __name__ == "__main__":
    result = verify_checkpoint_structure()
    
    if result['all_ready']:
        print(f"\n🚀 Ready to start MIMO Late Fusion training!")
        print(f"   Run: python combined_late_fusion.py")
    else:
        print(f"\n⚠️  Some directories were missing but have been created.")
        print(f"   You can now run: python combined_late_fusion.py")
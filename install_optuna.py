#!/usr/bin/env python3
"""
Installation script for Optuna and dependencies for hyperparameter optimization
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install required packages for hyperparameter optimization"""
    
    print("🔍 Installing Optuna and dependencies for hyperparameter optimization...")
    print("="*70)
    
    # List of packages to install
    packages = [
        "optuna>=3.0.0",
        "matplotlib>=3.5.0",  # For visualization plots
        "plotly>=5.0.0",      # For interactive plots
        "kaleido>=0.2.1",     # For static plot export
    ]
    
    failed_packages = []
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        if not install_package(package):
            failed_packages.append(package)
    
    print("\n" + "="*70)
    
    if failed_packages:
        print("❌ Some packages failed to install:")
        for package in failed_packages:
            print(f"   - {package}")
        print("\nPlease install them manually using:")
        print(f"   pip install {' '.join(failed_packages)}")
    else:
        print("🎉 All packages installed successfully!")
        print("\n🚀 You can now run hyperparameter optimization with:")
        print("   python train_video_improved.py optimize")
        print("   python train_video_improved.py optimize 30        # 30 trials")
        print("   python train_video_improved.py optimize 50 3600   # 50 trials, 1 hour timeout")
    
    print("="*70)

if __name__ == "__main__":
    main()

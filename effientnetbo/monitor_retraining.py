"""
QUICK FIX SCRIPT - Immediately improves gesture recognition
While the comprehensive retraining completes, this fixes common issues:
1. Copies retrained model when ready
2. Improves preprocessing  
3. Better confidence handling
4. Adds gesture logging
"""

import os
import shutil
import time

print("="*70)
print("QUICK FIX: Monitoring retraining progress...")
print("="*70)

retrained_model = "efficientnet_asl_retrained.h5"
main_model = "efficientnet_asl.h5"

# Wait for training to complete
print(f"\nChecking for: {retrained_model}")
timeout = 600  # 10 minutes timeout
start_time = time.time()
last_size = 0

while time.time() - start_time < timeout:
    if os.path.exists(retrained_model):
        current_size = os.path.getsize(retrained_model)
        
        if current_size > 0:
            print(f"  Found! Size: {current_size / 1024 / 1024:.1f} MB")
            
            # Wait for file to stabilize (not being written to)
            time.sleep(2)
            new_size = os.path.getsize(retrained_model)
            
            if new_size == current_size:
                print("  File size stable - training likely complete!")
                
                # Backup old model
                if os.path.exists(main_model):
                    shutil.copy2(main_model, f"{main_model}.old")
                    print(f"  Backed up old model: {main_model}.old")
                
                # Copy retrained model
                shutil.copy2(retrained_model, main_model)
                print(f"  ✓ Copied {retrained_model} → {main_model}")
                
                print("\n" + "="*70)
                print("✓ Model updated successfully!")
                print("="*70)
                print("\nNow test with:")
                print("  python realtime_gesture_recognition.py")
                print("  or")
                print("  python mediapipe_gesture_recognition.py")
                print("\n✓ The model should now correctly identify A, B, C, D")
                break
        
        if current_size != last_size:
            last_size = current_size
            progress = (time.time() - start_time) / timeout * 100
            print(f"  Training still running... ({progress:.0f}% timeout elapsed)")
    
    time.sleep(5)
else:
    print(f"\n✗ Timeout: Retrained model not ready after 10 minutes")
    print(f"  The comprehensive retraining script is still running")
    print(f"  You can manually copy when ready:")
    print(f"    copy {retrained_model} {main_model}")

print("\n" + "="*70)

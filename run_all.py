import subprocess

# Run YOLO script
subprocess.Popen([
    r"D:\safety_Glasses\yolo_env\Scripts\python.exe",
    "person_detection.py"
])

# Run Gaze script
subprocess.Popen([
    r"D:\safety_Glasses\gaze_env\Scripts\python.exe",
    "test_landmark.py"
])

print("Both systems running...")
input("Press ENTER to stop...")
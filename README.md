# Gesture-Controlled-Drone-Simulation
A computer-vision-based drone simulation that allows users to control a virtual drone using hand gestures captured via webcam. The system integrates gesture recognition, physics simulation, HUD visualization, and flight data logging.
📌 Features
🎥 Real-time hand tracking using MediaPipe
✋ Gesture-based flight commands:
Hover
Forward Thrust
Backward Thrust
Strafe Left
Landing
📊 Tactical HUD showing:
Velocity
Altitude
Battery percentage
🔋 Battery consumption model
🧭 World boundary & ground-level safety
🧾 CSV flight data logging
🖥️ Dual-screen view (Camera + Simulation)
🧠 Gesture Mapping
Gesture
Action
All fingers down
Landing
All fingers up
Hover
Two fingers up
Forward Thrust
One finger up
Backward Thrust
Three fingers up
Strafe Left
🛠️ Tech Stack
Python
OpenCV
MediaPipe
NumPy
▶️ How to Run
Copy code
Bash
pip install opencv-python mediapipe numpy
python main.py
📷 Ensure your webcam is connected before running.
📁 Output
Real-time drone simulation window
Gesture-controlled navigation
Flight log saved as:
Copy code

comprehensive_flight_log.csv
🎓 Educational Value
This project demonstrates:
Human-Computer Interaction (HCI)
Real-time Computer Vision
Physics-based simulation
State machines & telemetry systems
👤 Author
Sakshi pagare
Artificial intelligence Student

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
from collections import deque

# ------------------ CONFIGURATION ------------------
CAM_W, CAM_H = 640, 480
SIM_W, SIM_H = 640, 480
DRONE_SPEED = 0.09
WORLD_RADIUS = 6.0
GROUND_LEVEL = -4.8  # Physical floor limit
GESTURE_STABLE_FRAMES = 5
TRAIL_MAX_LEN = 50
LOG_FILE = "comprehensive_flight_log.csv"

# Energy Constants
IDLE_DRAIN = 0.008
THRUST_DRAIN = 0.035

# UI Style
FONT = cv2.FONT_HERSHEY_SIMPLEX
CLR_HUD = (0, 255, 120)    
CLR_WARN = (0, 40, 255)    
CLR_ACCENT = (255, 200, 0) 
CLR_BG = (10, 12, 15)      
CLR_GRID = (30, 35, 30)    

# ------------------ SYSTEM STATE ------------------
drone_x, drone_y, drone_z = 0.0, 0.0, 0.0
velocity = 0.0
prev_pos = (0, 0, 0)
drone_state = "SYSTEM_READY"
stable_gesture = "NONE"
gesture_history = deque(maxlen=GESTURE_STABLE_FRAMES)
flight_path = deque(maxlen=TRAIL_MAX_LEN)
battery_pct = 100.0

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

# Initialize Black Box CSV
with open(LOG_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "X", "Y", "Z", "Velocity", "Battery", "State", "Gesture"])

# ------------------ CORE LOGIC ------------------

def log_data():
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), round(drone_x, 2), round(drone_y, 2), round(drone_z, 2), 
                         round(velocity, 2), round(battery_pct, 1), drone_state, stable_gesture])

def classify_gesture(hand):
    lm = hand.landmark
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    up = [lm[t].y < lm[p].y for t, p in zip(tips, pips)]
    count = sum(up)
    
    if count == 0: return "LANDING"
    if count == 4: return "HOVER"
    if up[0] and up[1] and not up[2]: return "THRUST_FWD"
    if up[0] and not up[1]: return "THRUST_BWD"
    if count == 3: return "STRAFE_LEFT"
    return "STABLE"

def update_physics(hand_y_raw):
    global drone_x, drone_y, drone_z, drone_state, velocity, prev_pos, battery_pct
    
    # 1. Power Consumption
    drain = IDLE_DRAIN + (THRUST_DRAIN if stable_gesture.startswith("THRUST") else 0)
    battery_pct = max(0, battery_pct - drain)

    # 2. Movement & State Logic
    target_y = (0.5 - hand_y_raw) * 12 
    
    if stable_gesture == "LANDING":
        # Slow descent and horizontal braking
        drone_x *= 0.90
        drone_z *= 0.90
        if drone_y > GROUND_LEVEL:
            drone_y -= 0.05
            drone_state = "DESCENDING"
        else:
            drone_y = GROUND_LEVEL
            drone_state = "LANDED" # Final Landed state
    elif stable_gesture == "HOVER":
        # Maintain current position with high friction
        drone_x *= 0.98
        drone_z *= 0.98
        drone_y += (target_y - drone_y) * 0.05
        drone_state = "HOVERING"
    else:
        drone_state = "FLYING"
        drone_y += (target_y - drone_y) * 0.12 
        if stable_gesture == "THRUST_FWD": drone_z -= DRONE_SPEED
        elif stable_gesture == "THRUST_BWD": drone_z += DRONE_SPEED
        elif stable_gesture == "STRAFE_LEFT": drone_x -= DRONE_SPEED

    # Safety Floor
    if drone_y < GROUND_LEVEL: 
        drone_y = GROUND_LEVEL
        if drone_state != "DESCENDING": drone_state = "LANDED"

    # 3. Boundary & Velocity
    dist = math.sqrt(drone_x**2 + drone_z**2)
    if dist > WORLD_RADIUS:
        scale = WORLD_RADIUS / dist
        drone_x *= scale; drone_z *= scale

    curr_pos = (drone_x, drone_y, drone_z)
    velocity = math.sqrt(sum((a - b) ** 2 for a, b in zip(curr_pos, prev_pos))) * 110
    prev_pos = curr_pos
    
    log_data()

def draw_tactical_hud(sim):
    h, w, _ = sim.shape
    cx, cy = w // 2, h // 2
    sim[:] = CLR_BG
    
    # Grid
    ground_y, horizon_y = int(h * 0.85), int(h * 0.35)
    for i in range(-15, 16):
        cv2.line(sim, (cx + i*80, ground_y), (cx + int(i*20), horizon_y), CLR_GRID, 1)
    
    # Flight State Text
    state_color = CLR_HUD if drone_state in ["HOVERING", "LANDED"] else CLR_ACCENT
    cv2.putText(sim, f"STATE: {drone_state}", (30, 40), FONT, 0.7, state_color, 2)

    # Drone Projection
    sx = int(cx + (drone_x / WORLD_RADIUS) * (w * 0.45))
    z_norm = (drone_z + WORLD_RADIUS) / (2 * WORLD_RADIUS)
    sy = int(horizon_y + z_norm * (ground_y - horizon_y))
    sy_alt = int(sy - (drone_y * 16))

    flight_path.append((sx, sy_alt))
    for i in range(1, len(flight_path)):
        cv2.line(sim, flight_path[i-1], flight_path[i], (0, int(255*(i/TRAIL_MAX_LEN)), 150), 2)
    
    cv2.drawMarker(sim, (sx, sy_alt), CLR_ACCENT, cv2.MARKER_TILTED_CROSS, 25, 2)

    # Telemetry
    data = [("VEL", f"{velocity:.1f}"), ("ALT", f"{drone_y+5:.1f}"), ("BATT", f"{battery_pct:.1f}%")]
    for i, (label, val) in enumerate(data):
        cv2.putText(sim, f"{label}: {val}", (30, 80 + i*28), FONT, 0.5, CLR_HUD, 1)

def main():
    global stable_gesture
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h_y = 0.5
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h_y = hand.landmark[9].y
            gest = classify_gesture(hand)
            gesture_history.append(gest)
            if len(gesture_history) == GESTURE_STABLE_FRAMES:
                u = set(gesture_history)
                if len(u) == 1: stable_gesture = list(u)[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        update_physics(h_y)
        sim = np.zeros((SIM_H, SIM_W, 3), dtype=np.uint8)
        draw_tactical_hud(sim)

        cv2.imshow("NEURAL PILOT v3.0 - LANDING/HOVER UPDATE", np.hstack((cv2.resize(frame, (SIM_W, SIM_H)), sim)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

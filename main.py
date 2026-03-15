import cv2
import numpy as np
import json
import os
import sys
import argparse

# --- RESOLUTION SETTINGS ---
# Change these if you want 720p (1280x720) or 4K (3840x2160)
FINAL_W = 1920
FINAL_H = 1080
# Since the image is rotated 90 degrees, the intermediate dimensions are swapped
ROT_W = FINAL_H
ROT_H = FINAL_W

# --- PATH & CONFIG ---
BASE_PATH = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_PATH, "config.json")

DEFAULT_CONFIG = {
    # Default keystone points scaled to the new FINAL_W and FINAL_H
    "points": [[100, 100], [FINAL_W - 100, 100], [FINAL_W - 100, FINAL_H - 100], [100, FINAL_H - 100]],
    "zoom_roi": [0, 0, ROT_W, ROT_H], 
    "exposure": -5,
    "camera_id": 0
}

def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f: 
                return json.load(f)
    except: 
        pass
    return DEFAULT_CONFIG.copy()

def save_config(data):
    with open(CONFIG_FILE, 'w') as f: 
        json.dump(data, f, indent=4)

def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   
    rect[2] = pts[np.argmax(s)]   
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
    return rect

def get_bright_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return None, thresh

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 3000: 
        return None, thresh
    
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.05 * peri, True)

    if len(approx) == 4:
        return order_points(approx), thresh
    return None, thresh

# --- GLOBALS ---
config = load_config()
selected_point = -1
auto_track = False
drawing_zoom = False
zoom_start_pt = (0, 0)
save_tick = 0
calib_mode = False 

# --- MOUSE LOGIC ---
def mouse_zoom(event, x, y, flags, param):
    global config, drawing_zoom, zoom_start_pt
    if event == cv2.EVENT_RBUTTONDOWN:
        drawing_zoom = True
        zoom_start_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing_zoom:
        ix, iy = zoom_start_pt
        config["zoom_roi"] = [min(ix, x), min(iy, y), max(20, abs(x - ix)), max(20, abs(y - iy))]
    elif event == cv2.EVENT_RBUTTONUP:
        drawing_zoom = False

def mouse_keystone(event, x, y, flags, param):
    global config, selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        dists = [np.linalg.norm(np.array([x,y]) - np.array(p)) for p in config["points"]]
        selected_point = np.argmin(dists)
        config["points"][selected_point] = [x, y]
    elif event == cv2.EVENT_MOUSEMOVE and selected_point != -1:
        config["points"][selected_point] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_point = -1

def main():
    global config, auto_track, save_tick, calib_mode
    
    # Parse CLI Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--calib", action="store_true", help="Start in calibration mode")
    args = parser.parse_args()
    if args.calib: calib_mode = True

    cap = cv2.VideoCapture(config["camera_id"], cv2.CAP_DSHOW)
    
    # Request higher resolution from the camera hardware
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FINAL_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FINAL_H)
    cap.set(cv2.CAP_PROP_EXPOSURE, config["exposure"])
    
    cv2.namedWindow("FINAL_VIEW", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FINAL_VIEW", FINAL_W, FINAL_H)

    def update_window_visibility():
        if calib_mode:
            cv2.namedWindow("ZOOM_SELECTOR", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ZOOM_SELECTOR", ROT_W, ROT_H)
            cv2.setMouseCallback("ZOOM_SELECTOR", mouse_zoom)
            
            cv2.namedWindow("KEYSTONE_DEBUG", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("KEYSTONE_DEBUG", FINAL_W, FINAL_H)
            cv2.setMouseCallback("KEYSTONE_DEBUG", mouse_keystone)
        else:
            try:
                cv2.destroyWindow("ZOOM_SELECTOR")
                cv2.destroyWindow("KEYSTONE_DEBUG")
            except: 
                pass

    update_window_visibility()

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, (ROT_W, ROT_H)) 
        
        zx, zy, zw, zh = config["zoom_roi"]
        zoomed = frame[max(0, zy):min(ROT_H, zy+zh), max(0, zx):min(ROT_W, zx+zw)]
        if zoomed.size == 0: 
            zoomed = frame
        k_input = cv2.resize(zoomed, (FINAL_W, FINAL_H))

        # 1. Tracking and UI Processing
        new_pts, thresh_raw = get_bright_corners(k_input)
        
        if auto_track and new_pts is not None:
            curr = np.array(config["points"], dtype="float32")
            config["points"] = (curr * 0.8 + new_pts * 0.2).tolist()

        # 2. Final Warp (Always show)
        try:
            src = np.float32(config["points"])
            dst = np.float32([[0, 0], [FINAL_W, 0], [FINAL_W, FINAL_H], [0, FINAL_H]])
            M = cv2.getPerspectiveTransform(src, dst)
            final = cv2.warpPerspective(k_input, M, (FINAL_W, FINAL_H))
            cv2.imshow("FINAL_VIEW", final)
        except: 
            pass

        # 3. Handle Calibration Windows
        if calib_mode:
            debug_view = cv2.cvtColor(thresh_raw, cv2.COLOR_GRAY2BGR)
            pts_draw = np.array(config["points"], np.int32)
            cv2.polylines(debug_view, [pts_draw], True, (255, 0, 0), 2)
            for p in config["points"]: 
                cv2.circle(debug_view, (int(p[0]), int(p[1])), 8, (0, 255, 0), -1)
            
            status = f"AUTO: {'ON' if auto_track else 'OFF'}"
            cv2.putText(debug_view, status, (10, 30), 1, 1.5, (0, 0, 255), 2)
            
            if save_tick > 0:
                cv2.putText(debug_view, "SAVED!", (FINAL_W // 2 - 100, FINAL_H // 2), 1, 3.0, (0, 255, 0), 4)
                save_tick -= 1
            cv2.imshow("KEYSTONE_DEBUG", debug_view)

            z_ui = frame.copy()
            cv2.rectangle(z_ui, (zx, zy), (zx+zw, zy+zh), (0, 255, 255), 2)
            cv2.imshow("ZOOM_SELECTOR", z_ui)

        # 4. Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            save_config(config)
            break
        elif key == ord('k'): # TOGGLE CALIBRATION
            calib_mode = not calib_mode
            update_window_visibility()
        elif key == ord('c'): 
            save_config(config)
            save_tick = 30 
        elif key == ord('a'): 
            auto_track = not auto_track
        elif key == ord('r'): 
            config["zoom_roi"] = [0, 0, ROT_W, ROT_H]
        elif key == ord('='): 
            config["exposure"] += 1
            cap.set(cv2.CAP_PROP_EXPOSURE, config["exposure"])
        elif key == ord('-'):
            config["exposure"] -= 1
            cap.set(cv2.CAP_PROP_EXPOSURE, config["exposure"])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
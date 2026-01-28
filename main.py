import cv2
import numpy as np
import argparse
import json
import os
import sys

# --- EXE PATH LOGIC ---
if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(BASE_PATH, "config.json")

DEFAULT_CONFIG = {
    "points": [[0, 0], [640, 0], [640, 480], [0, 480]],
    "zoom_roi": [0, 0, 640, 480], 
    "exposure": -6,
    "camera_id": 1
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except: return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(data):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Interaction Globals
selected_point = -1
drawing_zoom = False
zoom_start_pt = (0,0)

def mouse_event_zoom(event, x, y, flags, param):
    global config, drawing_zoom, zoom_start_pt
    if event == cv2.EVENT_RBUTTONDOWN:
        drawing_zoom = True
        zoom_start_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing_zoom:
        ix, iy = zoom_start_pt
        config["zoom_roi"] = [min(ix, x), min(iy, y), max(20, abs(x - ix)), max(20, abs(y - iy))]
    elif event == cv2.EVENT_RBUTTONUP:
        drawing_zoom = False

def mouse_event_keystone(event, x, y, flags, param):
    global config, selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, p in enumerate(config["points"]):
            if abs(x - p[0]) < 20 and abs(y - p[1]) < 20:
                selected_point = i
    elif event == cv2.EVENT_MOUSEMOVE and selected_point != -1:
        config["points"][selected_point] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_point = -1

def main():
    global config
    config = load_config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib', action='store_true')
    args = parser.parse_args()

    cap = cv2.VideoCapture(config["camera_id"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, config["exposure"])

    main_win = "REAR_VIEW"
    cv2.namedWindow(main_win)

    if args.calib:
        cv2.namedWindow("ZOOM_SELECTOR")
        cv2.namedWindow("KEYSTONE_ADJUST")
        cv2.setMouseCallback("ZOOM_SELECTOR", mouse_event_zoom)
        cv2.setMouseCallback("KEYSTONE_ADJUST", mouse_event_keystone)

    while True:
        # Check if the main window was closed via the "X" button
        # On Windows, a closed window returns -1.0
        if cv2.getWindowProperty(main_win, cv2.WND_PROP_VISIBLE) < 1:
            break

        ret, raw_frame = cap.read()
        if not ret: 
            break

        full_frame = cv2.resize(raw_frame, (640, 480))

        # 1. APPLY ZOOM
        zx, zy, zw, zh = config["zoom_roi"]
        zx, zy = max(0, min(zx, 630)), max(0, min(zy, 470))
        zw, zh = max(10, min(zw, 640-zx)), max(10, min(zh, 480-zy))
        zoomed_area = full_frame[zy:zy+zh, zx:zx+zw]
        
        if zoomed_area.size == 0:
            keystone_input = full_frame
        else:
            keystone_input = cv2.resize(zoomed_area, (640, 480))

        # 2. APPLY KEYSTONE
        src_pts = np.float32(config["points"])
        dst_pts = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        final_view = cv2.warpPerspective(keystone_input, matrix, (640, 480))

        # 4. SHOW WINDOWS
        cv2.imshow(main_win, final_view)

        if args.calib:
            z_ui = full_frame.copy()
            cv2.rectangle(z_ui, (zx, zy), (zx+zw, zy+zh), (0, 255, 255), 2)
            cv2.imshow("ZOOM_SELECTOR", z_ui)

            k_ui = keystone_input.copy()
            pts_arr = np.array(config["points"], np.int32)
            cv2.polylines(k_ui, [pts_arr], True, (255, 0, 0), 2)
            for p in config["points"]:
                cv2.circle(k_ui, (int(p[0]), int(p[1])), 6, (0, 255, 0), -1)
            cv2.imshow("KEYSTONE_ADJUST", k_ui)

        # 5. KEYBOARD INPUT
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            config["zoom_roi"] = [0, 0, 640, 480]
            config["points"] = [[0, 0], [640, 0], [640, 480], [0, 480]]
        elif key == ord('=') or key == ord('+'):
            config["exposure"] += 1
            cap.set(cv2.CAP_PROP_EXPOSURE, config["exposure"])
        elif key == ord('-'):
            config["exposure"] -= 1
            cap.set(cv2.CAP_PROP_EXPOSURE, config["exposure"])

    # SAVE AND CLEANUP (Happens after loop breaks via Q or X)
    save_config(config)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
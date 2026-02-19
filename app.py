import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from scipy.interpolate import splprep, splev
import os
import requests

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="AI Beauty Lab: Commercial V10")

# --- 1. MATH ENGINE ---
def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1/2.4)) - 0.055)
    return (y * 255.0 + 0.5).astype(np.uint8)

def hex_to_rgb01(hex_color):
    s = hex_color.lstrip("#")
    return np.array([int(s[i:i+2], 16) for i in (0,2,4)], dtype=np.float32) / 255.0

def blend_linear_soft_light(base_img, mask, color_hex, opacity):
    base_lin = srgb_to_linear(base_img.astype(np.float32) / 255.0)
    tgt_rgb = hex_to_rgb01(color_hex)
    tgt_lin = srgb_to_linear(tgt_rgb)
    blend_layer = np.ones_like(base_lin) * tgt_lin
    
    res_lin = np.where(blend_layer <= 0.5,
                       2 * base_lin * blend_layer + base_lin**2 * (1 - 2*blend_layer),
                       2 * base_lin * (1 - blend_layer) + np.sqrt(np.clip(base_lin, 0, 1)) * (2*blend_layer - 1))
    
    res_srgb = linear_to_srgb(res_lin)
    alpha = mask[:, :, None] * opacity
    final = res_srgb * alpha + base_img * (1 - alpha)
    return final.astype(np.uint8)

def get_smooth_polygon_mask(h, w, landmarks, indices):
    raw_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    if len(raw_points) < 3: return np.zeros((h, w), dtype=np.float32)
    
    pts = np.array(raw_points)
    _, idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    
    if len(pts) < 4: return np.zeros((h, w), dtype=np.float32)

    try:
        pts = np.vstack([pts, pts[0]])
        tck, _ = splprep(pts.T, s=0, per=True) 
        u_new = np.linspace(0, 1, 100)
        x_new, y_new = splev(u_new, tck)
        smooth_pts = np.stack([x_new, y_new], axis=1).astype(np.int32)
    except:
        smooth_pts = pts.astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [smooth_pts], 255)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask.astype(np.float32) / 255.0

# --- 2. MAKEUP LOGIC ---
def apply_lipstick(img, landmarks, color, opacity):
    h, w = img.shape[:2]
    ul_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291] + [308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
    ll_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291] + [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
    
    mask_u = get_smooth_polygon_mask(h, w, landmarks, ul_indices)
    mask_l = get_smooth_polygon_mask(h, w, landmarks, ll_indices)
    return blend_linear_soft_light(img, np.clip(mask_u + mask_l, 0, 1), color, opacity)

def apply_eyeliner(img, landmarks, color, opacity):
    h, w = img.shape[:2]
    overlay = img.copy()
    col_bgr = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    
    for idxs in [[33, 7, 163, 144, 145, 153, 154, 155, 133], [362, 382, 381, 380, 374, 373, 390, 249, 263]]:
        pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs], dtype=np.int32)
        try:
            tck, _ = splprep(pts.T, s=0, k=2)
            x, y = splev(np.linspace(0, 1, 50), tck)
            smooth_pts = np.stack([x, y], axis=1).astype(np.int32)
        except:
            smooth_pts = pts
        cv2.polylines(overlay, [smooth_pts], False, col_bgr, 2, cv2.LINE_AA)
        
    return cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)

def apply_blush(img, landmarks, color, opacity):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for idxs in [[116, 117, 118, 100, 126, 209], [345, 346, 347, 329, 355, 429]]:
        cx = int(np.mean([landmarks[i].x for i in idxs]) * w)
        cy = int(np.mean([landmarks[i].y for i in idxs]) * h)
        blob = np.zeros((h, w), dtype=np.float32)
        cv2.circle(blob, (cx, cy), int(h * 0.08), 1.0, -1)
        mask = np.maximum(mask, cv2.GaussianBlur(blob, (99, 99), 0))
    return blend_linear_soft_light(img, mask, color, opacity)

def apply_eyeshadow(img, landmarks, color, opacity):
    h, w = img.shape[:2]
    mask = get_smooth_polygon_mask(h, w, landmarks, [226, 247, 30, 29, 27, 28, 56, 190]) + \
           get_smooth_polygon_mask(h, w, landmarks, [463, 414, 286, 258, 257, 259, 260, 467])
    return blend_linear_soft_light(img, cv2.GaussianBlur(np.clip(mask, 0, 1), (15, 15), 0), color, opacity)

def apply_foundation(img, landmarks, color, opacity):
    h, w = img.shape[:2]
    mask = get_smooth_polygon_mask(h, w, landmarks, [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109])
    
    eyes = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7] + [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
    lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    
    sub_mask = get_smooth_polygon_mask(h, w, landmarks, eyes) + get_smooth_polygon_mask(h, w, landmarks, lips)
    sub_mask = cv2.dilate(sub_mask, np.ones((10,10)), iterations=1)
    mask = cv2.GaussianBlur(np.maximum(mask - sub_mask, 0), (31, 31), 0)
    
    smooth_img = cv2.bilateralFilter(img, 15, 75, 75)
    colored_img = blend_linear_soft_light(smooth_img, np.ones((h, w), dtype=np.float32), color, 0.5)
    
    alpha = mask[:, :, None] * opacity
    return (colored_img * alpha + img * (1 - alpha)).astype(np.uint8)

# --- 3. MAIN APP ---
st.title("AI Makeup Lab (Physics Based)")

# Initialize FaceMesh - WE DO NOT NEED TO DOWNLOAD MODELS for FaceMesh, it is built-in.
# We REMOVED the Hair Segmenter entirely to simplify and fix the errors.
try:
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    )
except Exception as e:
    st.error(f"Error loading MediaPipe: {e}")
    st.stop()

with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Image", type=['jpg', 'png', 'jpeg'])
    
    st.header("Controls")
    l_en = st.checkbox("Lipstick", True)
    l_col = st.color_picker("Lip Color", "#D81B60")
    l_op = st.slider("Lip Opacity", 0.0, 1.0, 0.6)
    
    b_en = st.checkbox("Blush", False)
    b_col = st.color_picker("Blush Color", "#FF9999")
    b_op = st.slider("Blush Opacity", 0.0, 1.0, 0.4)
    
    es_en = st.checkbox("Eyeshadow", False)
    es_col = st.color_picker("Shadow Color", "#4A148C")
    es_op = st.slider("Shadow Opacity", 0.0, 1.0, 0.5)
    
    el_en = st.checkbox("Eyeliner", False)
    el_col = st.color_picker("Liner Color", "#000000")
    el_op = st.slider("Liner Opacity", 0.0, 1.0, 0.8)

    f_en = st.checkbox("Foundation", False)
    f_col = st.color_picker("Foundation Tone", "#E8CFA9")
    f_op = st.slider("Foundation Opacity", 0.0, 1.0, 0.5)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(img_rgb, use_container_width=True)
            
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            res = img_rgb.copy()
            
            if f_en: res = apply_foundation(res, landmarks, f_col, f_op)
            if b_en: res = apply_blush(res, landmarks, b_col, b_op)
            if es_en: res = apply_eyeshadow(res, landmarks, es_col, es_op)
            if l_en: res = apply_lipstick(res, landmarks, l_col, l_op)
            if el_en: res = apply_eyeliner(res, landmarks, el_col, el_op)

            with col2:
                st.subheader("Result")
                st.image(res, use_container_width=True)
        else:
            st.error("No face detected.")
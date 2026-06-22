from __future__ import annotations

import cv2

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]

    for a, b in connections:
        ax = int(hand_landmarks[a].x * w)
        ay = int(hand_landmarks[a].y * h)
        bx = int(hand_landmarks[b].x * w)
        by = int(hand_landmarks[b].y * h)

        cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)

    for lm in hand_landmarks:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)


def draw_scanner_overlay(
    image,
    predicted_letter: str,
    raw_prediction: str,
    confidence: float,
    margin: float,
    hand_count: int,
):
    h, w, _ = image.shape

    yellow = (66, 188, 245)  # BGR untuk kuning/orange
    green = (90, 220, 90)
    white = (245, 245, 245)
    dark = (18, 22, 30)

    # Area scanner
    x1 = int(w * 0.14)
    y1 = int(h * 0.20)
    x2 = int(w * 0.86)
    y2 = int(h * 0.78)

    corner_len = int(w * 0.12)
    thickness = 14

    # 4 corner scanner
    cv2.line(image, (x1, y1), (x1 + corner_len, y1), yellow, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_len), yellow, thickness)

    cv2.line(image, (x2, y1), (x2 - corner_len, y1), yellow, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_len), yellow, thickness)

    cv2.line(image, (x1, y2), (x1 + corner_len, y2), yellow, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - corner_len), yellow, thickness)

    cv2.line(image, (x2, y2), (x2 - corner_len, y2), yellow, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - corner_len), yellow, thickness)

    # Teks tengah
    if hand_count == 0:
        # text = "Pratinjau kamera"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = h // 2
        cv2.putText(
            image,
            # text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 180, 180),
            2,
        )

    # Status pill bawah kiri
    overlay = image.copy()
    pill_x1 = 22
    pill_y1 = h - 58
    pill_x2 = 245
    pill_y2 = h - 18

    cv2.rectangle(
        overlay,
        (pill_x1, pill_y1),
        (pill_x2, pill_y2),
        dark,
        -1,
    )

    image = cv2.addWeighted(overlay, 0.70, image, 0.30, 0)

    cv2.circle(image, (pill_x1 + 24, pill_y1 + 20), 6, green, -1)

    status_text = f"{hand_count} tangan terdeteksi"
    cv2.putText(
        image,
        status_text,
        (pill_x1 + 42, pill_y1 + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        white,
        2,
    )

    return image


def center_crop_to_4_3(image):
    h, w, _ = image.shape
    target_aspect = 4 / 3
    current_aspect = w / h

    if current_aspect > target_aspect:
        # Frame terlalu lebar, crop kiri-kanan
        new_w = int(h * target_aspect)
        x1 = (w - new_w) // 2
        cropped = image[:, x1:x1 + new_w]
    else:
        # Frame terlalu tinggi / portrait, crop atas-bawah
        new_h = int(w / target_aspect)
        y1 = (h - new_h) // 2
        cropped = image[y1:y1 + new_h, :]

    resized = cv2.resize(cropped, (640, 480))
    return resized



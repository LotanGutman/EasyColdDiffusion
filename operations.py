import cv2
import numpy as np


# --------------------------------------------------
# Helper: Drawing Organic/Jagged Crease Lines
# --------------------------------------------------
def _draw_organic_crease(mask, p1, p2, thickness):
    """Draws a line with subtle organic jitter instead of a rigid ruler line."""
    dist = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if dist < 2:
        return

    num_subpoints = max(3, dist // 15)
    xs = np.linspace(p1[0], p2[0], num_subpoints)
    ys = np.linspace(p1[1], p2[1], num_subpoints)

    points = []
    for i in range(num_subpoints):
        if i == 0 or i == num_subpoints - 1:
            points.append((int(xs[i]), int(ys[i])))
        else:
            jx = int(xs[i] + np.random.randint(-2, 3))
            jy = int(ys[i] + np.random.randint(-2, 3))
            points.append((jx, jy))

    for i in range(len(points) - 1):
        cv2.line(mask, points[i], points[i + 1], 1, thickness, cv2.LINE_AA)


# --------------------------------------------------
# Helper: Recursive Branching (מבנה עץ/ברק)
# --------------------------------------------------
def _draw_branching_crack(white_mask, dark_mask, start_pt, angle, length, thickness, depth=0):
    if length < 10 or thickness < 1 or depth > 3:
        return

    end_x = int(start_pt[0] + length * np.cos(angle))
    end_y = int(start_pt[1] + length * np.sin(angle))
    end_pt = (end_x, end_y)

    cv2.line(white_mask, start_pt, end_pt, 1, max(1, int(thickness)), cv2.LINE_AA)
    cv2.line(dark_mask, start_pt, end_pt, 1, max(1, int(thickness) + 2), cv2.LINE_AA)

    # המשך ענף מרכזי
    next_angle = angle + np.radians(np.random.uniform(-15, 15))
    next_length = length * np.random.uniform(0.6, 0.8)
    _draw_branching_crack(white_mask, dark_mask, end_pt, next_angle, next_length, thickness * 0.8, depth)

    # פיצול ענף צדדי (50% סיכוי)
    if np.random.rand() < 0.55:
        branch_dir = np.random.choice([-1, 1])
        branch_angle = angle + branch_dir * np.radians(np.random.uniform(25, 45))
        branch_length = length * np.random.uniform(0.4, 0.6)
        _draw_branching_crack(white_mask, dark_mask, end_pt, branch_angle, branch_length, thickness * 0.6, depth + 1)


# --------------------------------------------------
# Pattern 1: Minimal Realistic Paper Fold
# --------------------------------------------------
def _pattern_paper_folds(height, width, thickness):
    white_mask = np.zeros((height, width), dtype=np.float32)
    dark_mask = np.zeros((height, width), dtype=np.float32)

    y_start = int(height * np.random.uniform(0.35, 0.65))
    y_end = int(height * np.random.uniform(0.35, 0.65))
    p1 = (0, y_start)
    p2 = (width - 1, y_end)

    _draw_organic_crease(white_mask, p1, p2, thickness=1)
    _draw_organic_crease(dark_mask, p1, p2, thickness=thickness + 2)

    if np.random.rand() < 0.6:
        x_branch = int(width * np.random.uniform(0.3, 0.7))
        y_branch = int(y_start + (y_end - y_start) * (x_branch / width))

        branch_end = (x_branch + np.random.randint(-int(width * 0.2), int(width * 0.2)),
                      np.random.choice([0, height - 1]))

        _draw_organic_crease(white_mask, (x_branch, y_branch), branch_end, thickness=1)
        _draw_organic_crease(dark_mask, (x_branch, y_branch), branch_end, thickness=thickness + 1)

    return (white_mask, dark_mask), "paper_style"


# --------------------------------------------------
# Pattern 2: Glass Spiderweb Shatter (שבר זכוכית קורי עכביש)
# --------------------------------------------------
def _pattern_glass_shatter(height, width, thickness):
    white_mask = np.zeros((height, width), dtype=np.float32)
    dark_mask = np.zeros((height, width), dtype=np.float32)

    # 1. נקודת הפגיעה המרכזית (Impact Center)
    cx = int(width * np.random.uniform(0.2, 0.5))
    cy = int(height * np.random.uniform(0.2, 0.5))
    center = (cx, cy)

    # 2. קרניים רדיאליות (Radial Rays) יוצאות מהמרכז
    num_rays = np.random.randint(6, 10)
    ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    ray_angles += np.radians(np.random.uniform(-15, 15, size=num_rays))

    for angle in ray_angles:
        curr = center
        max_dist = np.random.uniform(0.3, 0.6) * np.hypot(width, height)
        steps = np.random.randint(4, 7)
        step_len = max_dist / steps

        for _ in range(steps):
            angle += np.radians(np.random.uniform(-10, 10))
            nx = int(curr[0] + step_len * np.cos(angle))
            ny = int(curr[1] + step_len * np.sin(angle))
            nxt = (nx, ny)

            # ציור הקרן (צל כהה + הילה לבנה של הברקת זכוכית)
            cv2.line(dark_mask, curr, nxt, 1, max(1, thickness), cv2.LINE_AA)
            cv2.line(white_mask, (curr[0] + 1, curr[1] + 1), (nxt[0] + 1, nxt[1] + 1), 1, max(1, thickness - 1),
                     cv2.LINE_AA)
            curr = nxt

    # 3. טבעות היקפיות (Concentric Web Rings) - יצירת מבנה הקורים
    num_rings = np.random.randint(2, 4)
    max_radius = np.random.uniform(0.15, 0.35) * min(width, height)

    for r_idx in range(1, num_rings + 1):
        radius = (r_idx / num_rings) * max_radius

        for i in range(num_rays):
            a1 = ray_angles[i]
            a2 = ray_angles[(i + 1) % num_rays]

            # 75% סיכוי לציור מקטע טבעת (כדי לשמור על מראה אקראי ולא סימטרי)
            if np.random.rand() < 0.75:
                p1 = (int(cx + radius * np.cos(a1)), int(cy + radius * np.sin(a1)))
                p2 = (int(cx + radius * np.cos(a2)), int(cy + radius * np.sin(a2)))

                mid_a = (a1 + a2) / 2
                mid_r = radius * np.random.uniform(0.85, 1.0)
                p_mid = (int(cx + mid_r * np.cos(mid_a)), int(cy + mid_r * np.sin(mid_a)))

                # ציור הקשת ב-2 מקטעים
                cv2.line(dark_mask, p1, p_mid, 1, max(1, thickness - 1), cv2.LINE_AA)
                cv2.line(dark_mask, p_mid, p2, 1, max(1, thickness - 1), cv2.LINE_AA)

                cv2.line(white_mask, (p1[0] + 1, p1[1] + 1), (p_mid[0] + 1, p_mid[1] + 1), 1, 1, cv2.LINE_AA)
                cv2.line(white_mask, (p_mid[0] + 1, p_mid[1] + 1), (p2[0] + 1, p2[1] + 1), 1, 1, cv2.LINE_AA)

    return (white_mask, dark_mask), "paper_style"


# --------------------------------------------------
# Pattern 3: Corner Tree / Lightning Shatter (תוקנה כפילות הזוויות)
# --------------------------------------------------
def _pattern_lightning_tree(height, width, thickness):
    white_mask = np.zeros((height, width), dtype=np.float32)
    dark_mask = np.zeros((height, width), dtype=np.float32)

    start_corner = np.random.choice(['top_left', 'bottom_left', 'top_right'])

    if start_corner == 'top_left':
        start_pt = (0, int(height * np.random.uniform(0.05, 0.25)))
        base_angle = np.radians(np.random.uniform(20, 50))
    elif start_corner == 'bottom_left':
        start_pt = (0, int(height * np.random.uniform(0.75, 0.95)))
        base_angle = np.radians(np.random.uniform(-50, -20))
    else:  # top_right
        start_pt = (width - 1, int(height * np.random.uniform(0.05, 0.25)))
        base_angle = np.radians(np.random.uniform(130, 160))  # <--- תוקן! היה כאן np.radians כפול

    initial_length = np.hypot(width, height) * 0.35
    _draw_branching_crack(white_mask, dark_mask, start_pt, base_angle, initial_length, thickness=max(2, thickness), depth=0)

    return (white_mask, dark_mask), "paper_style"


# --------------------------------------------------
# Pattern 4: Side Edge Tear
# --------------------------------------------------
def _pattern_edge_tear(height, width, thickness):
    white_mask = np.zeros((height, width), dtype=np.float32)
    dark_mask = np.zeros((height, width), dtype=np.float32)

    y_entry = int(height * np.random.uniform(0.3, 0.7))
    start_pt = (0, y_entry)

    p1 = (int(width * 0.25), y_entry + np.random.randint(-30, 30))
    p2 = (int(width * 0.45), p1[1] + np.random.randint(-40, 40))

    nodes = [start_pt, p1, p2]
    for i in range(len(nodes) - 1):
        cv2.line(white_mask, nodes[i], nodes[i + 1], 1, thickness, cv2.LINE_AA)
        cv2.line(dark_mask, nodes[i], nodes[i + 1], 1, thickness + 3, cv2.LINE_AA)

        if i > 0 and np.random.rand() < 0.7:
            sub_angle = np.radians(np.random.uniform(-60, 60))
            sub_end = (int(nodes[i][0] + 40 * np.cos(sub_angle)), int(nodes[i][1] + 40 * np.sin(sub_angle)))
            cv2.line(white_mask, nodes[i], sub_end, 1, 1, cv2.LINE_AA)
            cv2.line(dark_mask, nodes[i], sub_end, 1, 2, cv2.LINE_AA)

    return (white_mask, dark_mask), "paper_style"


# --------------------------------------------------
# New Pattern 5: Wide Crack / Emulsion Peeling (סדק רחב וקילוף אמולסיה)
# --------------------------------------------------
def _pattern_emulsion_peeling(height, width, thickness):
    """
    יוצרת סדק רחב המדמה קילוף/נשירה של שכבת הצבע וחשיפה של הנייר הלבן מתחתיה.
    """
    peel_mask = np.zeros((height, width), dtype=np.float32)

    # 1. יצירת מסלול מרכזי רחב
    y_entry = int(height * np.random.uniform(0.2, 0.8))
    start_pt = (0, y_entry) if np.random.rand() < 0.5 else (width - 1, y_entry)

    # נקודות מעבר בזיגזג
    p1 = (int(width * 0.3), y_entry + np.random.randint(-50, 50))
    p2 = (int(width * 0.6), p1[1] + np.random.randint(-60, 60))
    end_pt = (int(width * 0.85), p2[1] + np.random.randint(-40, 40))

    nodes = [start_pt, p1, p2, end_pt]

    # עובי בסיסי רחב מאוד לסדק (פי 3-5 מסדק רגיל)
    wide_thickness = max(8, thickness * 4)

    # ציור המסלול המרכזי בעובי משתנה
    for i in range(len(nodes) - 1):
        curr_thick = int(wide_thickness * np.random.uniform(0.7, 1.3))
        cv2.line(peel_mask, nodes[i], nodes[i + 1], 1.0, curr_thick, cv2.LINE_AA)

        # הוספת "איים" או סדקים היקפיים קטנים שנפלטו מהקרע המרכזי
        if np.random.rand() < 0.6:
            branch_angle = np.radians(np.random.uniform(-70, 70))
            branch_len = np.random.randint(20, 50)
            b_end = (int(nodes[i][0] + branch_len * np.cos(branch_angle)),
                     int(nodes[i][1] + branch_len * np.sin(branch_angle)))
            cv2.line(peel_mask, nodes[i], b_end, 1.0, max(2, curr_thick // 3), cv2.LINE_AA)

    # 2. הפיכת הקווים לצורה אורגנית ומחוספסת (עיוות דיסטורשן)
    noise = np.random.uniform(-2, 2, (height, width)).astype(np.float32)
    noise_blur = cv2.GaussianBlur(noise, (5, 5), 0)
    peel_mask = np.clip(peel_mask + noise_blur * 0.2 * peel_mask, 0, 1)

    # מחזיר מסיכה בודדת עם סגנון ייחודי "peel_style"
    return peel_mask, "peel_style"


def cracks(image, pattern_type='random', intensity=0.75, thickness=3):
    height, width = image.shape[:2]
    result = image.copy().astype(np.float32)

    patterns = {
        'paper': _pattern_paper_folds,
        'glass': _pattern_glass_shatter,
        'tree': _pattern_lightning_tree,
        'tear': _pattern_edge_tear
    }

    if pattern_type == 'random' or pattern_type not in patterns:
        selected_key = np.random.choice(list(patterns.keys()))
        pattern_func = patterns[selected_key]
    else:
        pattern_func = patterns[pattern_type]

    mask_data, style = pattern_func(height, width, thickness)

    # --------------------------------------------------
    # טיפול בסגנון הסדק הרחב (Peeling / Flaking)
    # --------------------------------------------------
    if style == "peel_style":
        peel_mask = mask_data

        # א. יצירת צל כהה (Shadow) מסביב לשולי הקילוף
        shadow_k = max(7, int(thickness * 3) | 1)
        shadow_blur = cv2.GaussianBlur(peel_mask, (shadow_k, shadow_k), 0)
        shadow_factor = 1.0 - (0.5 * intensity * shadow_blur)

        if len(image.shape) == 3:
            for c in range(3):
                result[:, :, c] *= shadow_factor
        else:
            result *= shadow_factor

        # ב. יצירת מרקם נייר לבן-שמנת בתוך השטח שנחשף
        paper_base = np.full_like(result, 240.0)  # צבע נייר בהיר

        # הוספת גרעיניות עדינה למרקם הנייר שנחשף
        paper_noise = np.random.normal(0, 8.0, result.shape).astype(np.float32)
        paper_texture = np.clip(paper_base + paper_noise, 200, 255)

        # ג. חיתוך חד לקצוות הקילוף (Thresholding רך לחדות)
        binary_peel = cv2.threshold(peel_mask, 0.2, 1.0, cv2.THRESH_BINARY)[1]
        binary_peel_smooth = cv2.GaussianBlur(binary_peel, (3, 3), 0)

        # ד. שילוב בין הנייר שנחשף לתמונה המקורית
        if len(image.shape) == 3:
            for c in range(3):
                mask_c = binary_peel_smooth
                result[:, :, c] = result[:, :, c] * (1.0 - mask_c) + paper_texture[:, :, c] * mask_c
        else:
            result = result * (1.0 - binary_peel_smooth) + paper_texture * binary_peel_smooth

    elif style == "paper_style":
        # ... (הקוד הקיים של paper_style)
        white_mask, dark_mask = mask_data
        k_white = max(3, int(thickness) | 1)
        k_dark = max(5, int(thickness * 2) | 1)

        white_blur = cv2.GaussianBlur(white_mask, (k_white, k_white), 0)
        dark_blur = cv2.GaussianBlur(dark_mask, (k_dark, k_dark), 0)

        shadow_factor = 1.0 - (0.30 * intensity * dark_blur)
        if len(image.shape) == 3:
            for c in range(3):
                result[:, :, c] *= shadow_factor
        else:
            result *= shadow_factor

        bright_add = 70.0 * intensity * white_blur
        if len(image.shape) == 3:
            for c in range(3):
                result[:, :, c] += bright_add
        else:
            result += bright_add

    else:  # glass
        # ... (הקוד הקיים של glass)
        crack_mask = mask_data
        k_glass = max(3, int(thickness) | 1)
        blur_crack = cv2.GaussianBlur(crack_mask, (k_glass, k_glass), 0)

        dark_factor = 1.0 - (0.7 * intensity * blur_crack)
        if len(image.shape) == 3:
            for c in range(3):
                result[:, :, c] *= dark_factor
        else:
            result *= dark_factor

    return np.clip(result, 0, 255).astype(np.uint8)




#מוסיף נקודות של אבק וללוך
def add_dust_and_flecks(image, num_flecks=25):
    """מוסיף גרגירי אבק זעירים ולכלוכים נקודתיים בצורה בטוחה לכל סוג תמונה."""
    result = image.copy()
    height, width = image.shape[:2]
    is_color = (len(image.shape) == 3)

    for _ in range(num_flecks):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        size = np.random.randint(1, 3)

        # המרה מפורשת ל-int כדי למנוע בעיות טיפוסים ב-OpenCV
        val = int(np.random.choice([0, 245]))

        # התאמת צבע/סקלר לפי מספר הערוצים בתמונה
        color = (val, val, val) if is_color else val

        cv2.circle(result, (x, y), size, color, -1)

    return result



#כתמי חלודה\תה
def add_stain(image, intensity=0.4):
    """מוסיף כתם מים/לחות אורגני בלתי-סימטרי לתמונה."""
    height, width = image.shape[:2]
    stain_mask = np.zeros((height, width), dtype=np.float32)

    # מיקום הכתם (בדרך כלל ליד השוליים)
    cx = int(width * np.random.uniform(0.1, 0.9))
    cy = int(height * np.random.uniform(0.1, 0.9))
    max_radius = int(min(width, height) * np.random.uniform(0.15, 0.3))

    # יצירת צורה אורגנית ולא עיגול מושלם
    num_points = 16
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = max_radius * np.random.uniform(0.6, 1.2, size=num_points)

    pts = []
    for a, r in zip(angles, radii):
        pts.append([int(cx + r * np.cos(a)), int(cy + r * np.sin(a))])

    cv2.fillPoly(stain_mask, [np.array(pts, np.int32)], 1.0)

    # טשטוש חזק לקצוות רכים מאוד
    blur_k = int(max_radius * 0.8) | 1
    stain_mask = cv2.GaussianBlur(stain_mask, (blur_k, blur_k), 0)

    # יצירת גוון חלודה/צהבהב (Sepia Tinting לכתם)
    result = image.copy().astype(np.float32)

    if len(image.shape) == 3:  # BGR
        # הורדת כחול (הצהבה) והוספת מעט אדום/ירוק
        result[:, :, 0] *= (1.0 - 0.4 * intensity * stain_mask)  # Blue
        result[:, :, 1] *= (1.0 - 0.15 * intensity * stain_mask)  # Green
        result[:, :, 2] *= (1.0 - 0.05 * intensity * stain_mask)  # Red
    else:
        result *= (1.0 - 0.25 * intensity * stain_mask)

    return np.clip(result, 0, 255).astype(np.uint8)



# --------------------------------------------------
# Blur # Applies a Gaussian blur to soften the image.
# --------------------------------------------------
def blur(image, sigma_ratio=0.005):
    max_dim = max(image.shape[:2])
    actual_sigma = max_dim * sigma_ratio
    return cv2.GaussianBlur(image, (0, 0), actual_sigma)


# --------------------------------------------------
# Film Grain (Adaptive)
# Generates resolution-independent film grain noise.
# --------------------------------------------------
def film_grain(image, strength=10, grain_scale_ratio=0.002):
    height, width = image.shape[:2]
    max_dim = max(height, width)

    scale = max(1, int(max_dim * grain_scale_ratio))

    small_h = max(1, height // scale)
    small_w = max(1, width // scale)

    shape = (small_h, small_w, image.shape[2]) if image.ndim == 3 else (small_h, small_w)
    noise_small = np.random.normal(0, strength, shape)

    noise = cv2.resize(noise_small, (width, height), interpolation=cv2.INTER_NEAREST)

    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


# --------------------------------------------------
# Sepia
# Applies a warm brown/yellow tone to the image.
# --------------------------------------------------
def sepia(image, amount=1.0):
    # מטריצת Sepia המותאמת ישירות לסדר BGR של OpenCV
    # ערוץ 0: B, ערוץ 1: G, ערוץ 2: R
    sepia_matrix = np.array([
        [0.131, 0.534, 0.272],  # Output Blue
        [0.168, 0.686, 0.349],  # Output Green
        [0.189, 0.769, 0.393]   # Output Red
    ], dtype=np.float32)

    image_float = image.astype(np.float32)
    sepia_image = cv2.transform(image_float, sepia_matrix)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    return cv2.addWeighted(image, 1 - amount, sepia_image, amount, 0)


# --------------------------------------------------
# Contrast
# Reduces contrast to create a faded appearance.
# --------------------------------------------------

def change_contrast(image, alpha=0.75, beta=15):

    result = alpha * image.astype(np.float32) + beta

    return np.clip(result, 0, 255).astype(np.uint8)


# --------------------------------------------------
# Gamma Correction
# Adjusts the brightness of the image using gamma.
# --------------------------------------------------

def gamma_correction(image, gamma=1.1):

    normalized = image.astype(np.float32) / 255.0

    corrected = normalized ** gamma

    return np.clip(
        corrected * 255,
        0,
        255
    ).astype(np.uint8)


# --------------------------------------------------
# Vignette
# Darkens the corners of the image.
# --------------------------------------------------

def vignette(image, strength=0.5):

    height, width = image.shape[:2]

    # Create coordinate grids
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)

    X, Y = np.meshgrid(x, y)

    # Calculate distance from the image center
    distance = np.sqrt(X**2 + Y**2)

    # Create a smooth vignette mask
    mask = 1 - strength * np.clip(
        distance / np.sqrt(2),
        0,
        1
    )

    # Apply the mask to all color channels
    result = image.astype(np.float32) * mask[:, :, np.newaxis]

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# --------------------------------------------------
# Desaturation
# Reduces color saturation using the HSV color space.
# --------------------------------------------------
def desaturation(image, factor=0.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    hsv_uint8 = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR)




def add_blur_stain(
    image,
    center=None,
    radius=40,
    blur_strength=50,
    brightness_factor=1.15,
    skip_probability=0.3,
):
    """מוסיפה כתם מטושטש ובהיר באזור ספציפי או אקראי בתמונה.

    פרמטרים:
      - image: תמונת הקלט
      - center: מיקום הכתם (x, y) בפיקסלים. אם None - יוגרל אקראית בתמונה.
      - radius: רדיוס הכתם בפיקסלים
      - blur_strength: עוצמת הטשטוש באזור הכתם (מספר אי-זוגי)
      - brightness_factor: פקטור הבהרה (למשל 1.15 = 15%+ בהירות, 1.0 = ללא
        שינוי)
      - skip_probability: הסיכוי שהפונקציה תדלג ולא תבצע כלום
    """
    # 0. בדיקת דילוג אקראי
    if np.random.rand() < skip_probability:
        return image.copy()

    height, width = image.shape[:2]
    result = image.astype(np.float32)

    # 1. קביעת מרכז הכתם (רנדומלי אם לא צוין)
    if center is None:
        margin_x = min(radius, width // 4)
        margin_y = min(radius, height // 4)
        center_x = np.random.randint(margin_x, width - margin_x)
        center_y = np.random.randint(margin_y, height - margin_y)
        center = (center_x, center_y)

    # 2. יצירת מסיכה מעגלית עם קצוות רכים (Soft Edge Mask)
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.circle(mask, center, radius, 1.0, -1)

    # טשטוש קצוות המסכה לקבלת מעבר הדרגתי (Feathering)
    blur_ksize = max(21, int(radius * 0.8) | 1)
    soft_mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

    # 3. יצירת גרסה מטושטשת ומובהרת של התמונה
    ksize = max(3, int(blur_strength) | 1)  # חובה מספר אי-זוגי
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0).astype(
        np.float32
    )

    # העלאת הבהירות באזור המטושטש (מואר/בוהק)
    blurred_image *= brightness_factor

    # 4. מיזוג הדרגתי
    soft_mask_3d = (
        np.expand_dims(soft_mask, axis=2)
        if len(image.shape) == 3
        else soft_mask
    )

    result = result * (1.0 - soft_mask_3d) + blurred_image * soft_mask_3d

    return np.clip(result, 0, 255).astype(np.uint8)





# --------------------------------------------------
# Crumple Effect (Realistic Folded & Creased Paper)
# --------------------------------------------------
def crumple_effect(image, num_creases=12, intensity=0.6):
    """
    Simulates a crumpled paper effect with light and shadow along fold lines.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    num_creases : int
        Number of major fold lines crossing the image.
    intensity : float
        Strength of the highlights and shadows (0.1 to 1.0).
    """
    height, width = image.shape[:2]

    # 1. יצירת מפת צל ואור (Shading Canvas)
    shading = np.zeros((height, width), dtype=np.float32)

    for _ in range(num_creases):
        # הגרלת קו קפל מקצה לקצה
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)

        # יצירת מסיכת קפל בודד
        crease_mask = np.zeros((height, width), dtype=np.float32)
        cv2.line(crease_mask, (x1, y1), (x2, y2), 1.0, thickness=2)

        # טשטוש הקו ליצירת המעבר הרך של הקיפול בנייר
        blur_size = int(max(width, height) * 0.03) | 1  # גודל אי-זוגי
        blurred_line = cv2.GaussianBlur(crease_mask, (blur_size, blur_size), 0)

        # חישוב נגזרת (Gradient) ליצירת צד אחד מואר וצד אחד מוצל לאורך הקפל
        sobel_x = cv2.Sobel(blurred_line, cv2.CV_32F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(blurred_line, cv2.CV_32F, 0, 1, ksize=5)

        fold_shading = sobel_x + sobel_y
        shading += fold_shading

    # 2. נורמליזציה של מפת התאורה לטווח של [1.0 - intensity, 1.0 + intensity]
    if np.max(np.abs(shading)) > 0:
        shading = shading / np.max(np.abs(shading))

    # ערכים חיוביים יאירו, ערכים שליליים יחשיכו
    lighting_factor = 1.0 + (shading * intensity * 0.5)

    # 3. החלת התאורה על התמונה
    result = image.astype(np.float32)

    if len(image.shape) == 3:  # תמונת צבע (RGB / BGR)
        for c in range(3):
            result[:, :, c] *= lighting_factor
    else:  # תמונה בשחור-לבן
        result *= lighting_factor

    return np.clip(result, 0, 255).astype(np.uint8)

import cv2
import numpy as np

def add_random_heavy_tear(image, max_tear_width=5, add_dark_edge=True):
    """יוצרת קרע נייר ריאליסטי - מותאם לרזולוציית 128x128.

    שיפורים:
    - תמיכה בקרע חוצה (Full-length) או קרע חלקי שנעצר באמצע התמונה (Partial
    Tear).
    - דיקוק הדרגתי (Tapering) ככל שהקרע מתרחק מקצה ההתחלה.
    - אפקט שוליים מפותלים/מקופלים (Curled Flaps) עם הצללה ותאורה בהירה.
    """
    skip_probability = 0.6
    if np.random.rand() < skip_probability:
        return image.copy()

    height, width = image.shape[:2]
    img_diag = int(np.hypot(height, width))
    is_color = len(image.shape) == 3

    # --------------------------------------------------
    # 1. הגדרת סוג הקרע והפרמטרים הגיאומטריים (מותאם ל-128x128)
    # --------------------------------------------------
    is_partial_tear = np.random.rand() < 0.9

    angle_deg = np.random.uniform(0, 360)
    angle_rad = np.radians(angle_deg)

    # הקטנת עובי הקרע וההיסט הצידי
    tear_width = np.random.randint(2, max(3, max_tear_width + 1))
    shift_amount = np.random.randint(2, 5)

    dir_x, dir_y = np.cos(angle_rad), np.sin(angle_rad)
    perp_x, perp_y = -dir_y, dir_x

    cx = width / 2 + np.random.uniform(-width * 0.15, width * 0.15)
    cy = height / 2 + np.random.uniform(-height * 0.15, height * 0.15)

    if is_partial_tear:
        # קרע שמתחיל בקצה אחד ונעצר באזור המרכז
        t_start = -img_diag / 2
        t_end = np.random.uniform(-img_diag * 0.05, img_diag * 0.15)
        num_steps = np.random.randint(15, 25)
    else:
        # קרע מלא מקצה לקצה
        t_start = -img_diag / 2
        t_end = img_diag / 2
        num_steps = np.random.randint(20, 35)

    t_vals = np.linspace(t_start, t_end, num_steps)

    # --------------------------------------------------
    # 2. חישוב מסלול מרכזי ועובי משתנה (מתינון הזיגזג)
    # --------------------------------------------------
    # צמצום הזיגזג מ-[-14, 14] ל-[-4, 4] למסלול ישר ועדין יותר
    y_offsets = np.random.uniform(-4, 4, size=num_steps)
    y_offsets[0] = 0
    if not is_partial_tear:
        y_offsets[-1] = 0

    center_points = []
    width_factors = []

    for i in range(num_steps):
        t = t_vals[i]
        off = y_offsets[i]
        px = int(cx + t * dir_x + off * perp_x)
        py = int(cy + t * dir_y + off * perp_y)
        center_points.append((px, py))

        # חישוב הפחתת עובי (Tapering) לקרע חלקי
        if is_partial_tear:
            progress = i / (num_steps - 1)  # 0.0 בתחילה, 1.0 בסוף
            factor = (1.0 - progress) ** 1.2
        else:
            factor = 1.0
        width_factors.append(factor)

    # --------------------------------------------------
    # 3. הסטת התמונה בצד אחד של השבר
    # --------------------------------------------------
    split_mask = np.zeros((height, width), dtype=np.uint8)
    poly_pts = [p for p in center_points]

    far_p1 = [
        int(center_points[-1][0] + img_diag * perp_x),
        int(center_points[-1][1] + img_diag * perp_y),
    ]
    far_p0 = [
        int(center_points[0][0] + img_diag * perp_x),
        int(center_points[0][1] + img_diag * perp_y),
    ]
    poly_pts.extend([far_p1, far_p0])

    cv2.fillPoly(split_mask, [np.array(poly_pts, dtype=np.int32)], 255)

    shift_x = int(shift_amount * perp_x)
    shift_y = int(shift_amount * perp_y)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(
        image, M, (width, height), borderMode=cv2.BORDER_REFLECT
    )

    result = image.copy()
    if is_color:
        for c in range(3):
            result[:, :, c] = np.where(
                split_mask == 255, shifted_image[:, :, c], result[:, :, c]
            )
    else:
        result = np.where(split_mask == 255, shifted_image, result)

    # --------------------------------------------------
    # 4. בניית מצולע הקרע (Tear Polygon)
    # --------------------------------------------------
    top_edge = []
    bottom_edge = []

    for i in range(num_steps):
        px, py = center_points[i]
        f = width_factors[i]

        w_top = max(0.8, tear_width * f * np.random.uniform(0.5, 1.1))
        w_bot = max(0.8, tear_width * f * np.random.uniform(0.5, 1.1))

        p_top = (int(px + w_top * perp_x), int(py + w_top * perp_y))
        p_bot = (int(px - w_bot * perp_x), int(py - w_bot * perp_y))

        top_edge.append(p_top)
        bottom_edge.append(p_bot)

    tear_poly = np.array(top_edge + bottom_edge[::-1], dtype=np.int32)
    tear_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(tear_mask, [tear_poly], 255)

    # --------------------------------------------------
    # 5. הצללת עומק היקפית (רדיוס מותאם ל-128x128)
    # --------------------------------------------------
    res_float = result.astype(np.float32)
    shadow_blur = (
        cv2.GaussianBlur(tear_mask.astype(np.float32), (9, 9), 0) / 255.0
    )
    shadow_factor = 1.0 - (0.35 * shadow_blur)

    if is_color:
        for c in range(3):
            res_float[:, :, c] *= shadow_factor
    else:
        res_float *= shadow_factor

    # --------------------------------------------------
    # 6. מילוי במרקם הנייר שנחשף
    # --------------------------------------------------
    paper_base = np.full_like(image, (238, 240, 242), dtype=np.uint8)
    paper_noise = np.random.normal(0, 4, image.shape).astype(np.int16)
    paper_textured = np.clip(
        paper_base.astype(np.int16) + paper_noise, 0, 255
    ).astype(np.uint8)

    mask_bool = tear_mask == 255
    if is_color:
        for c in range(3):
            res_float[:, :, c][mask_bool] = paper_textured[:, :, c][mask_bool]
    else:
        res_float[mask_bool] = paper_textured[mask_bool]

    # --------------------------------------------------
    # 7. אפקט השוליים המפותלים/מקופלים (Curled Paper Flaps)
    # --------------------------------------------------
    flap_mask = np.zeros((height, width), dtype=np.float32)
    cv2.polylines(
        flap_mask,
        [np.array(top_edge, dtype=np.int32)],
        isClosed=False,
        color=1.0,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.polylines(
        flap_mask,
        [np.array(bottom_edge, dtype=np.int32)],
        isClosed=False,
        color=1.0,
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    flap_light = cv2.GaussianBlur(flap_mask, (5, 5), 0)
    flap_shadow = cv2.GaussianBlur(flap_mask, (9, 9), 0)

    # הוספת תאורה בהירה על הנייר המקופל והצללה רכה מתחתיו
    bright_boost = 1.0 + (0.30 * flap_light)
    shadow_decay = 1.0 - (0.20 * flap_shadow)

    if is_color:
        for c in range(3):
            res_float[:, :, c] = res_float[:, :, c] * shadow_decay * bright_boost
    else:
        res_float = res_float * shadow_decay * bright_boost

    # --------------------------------------------------
    # 8. קו מתאר כהה וחד לשולי הנייר (Dark Edge Outline)
    # --------------------------------------------------
    if add_dark_edge:
        edge_mask = np.zeros((height, width), dtype=np.float32)
        cv2.polylines(
            edge_mask,
            [tear_poly],
            isClosed=not is_partial_tear,
            color=1.0,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        edge_dark_factor = 1.0 - (0.50 * edge_mask)
        if is_color:
            for c in range(3):
                res_float[:, :, c] *= edge_dark_factor
        else:
            res_float *= edge_dark_factor

    return np.clip(res_float, 0, 255).astype(np.uint8)



def add_spilled_stain(
    image, stain_type="coffee", intensity=0.95, skip_probability=0.2
):
    """מוסיפה כתם נוזלי שנשפך/נמרח (כמו קפה, תה או שמן) עם אפקט Coffee-Ring

    ונתזים היקפיים.

    Parameters
    ----------
    image : np.ndarray
        תמונת הקלט.
    stain_type : str
        סוג הכתם: 'coffee' (קפה/חום), 'tea' (צהבהב/חלודה), 'grease' (שומני/כהה).
    intensity : float
        עוצמת הכתם וההכהיה (בין 0.1 ל-1.0).
    skip_probability : float
        הסתברות לדילוג ללא שינוי.
    """
    if np.random.rand() < skip_probability:
        return image.copy()

    height, width = image.shape[:2]
    is_color = len(image.shape) == 3
    result = image.astype(np.float32)

    # 1. הגדרת מיקום ורדיוס בסיסי
    cx = int(width * np.random.uniform(0.2, 0.8))
    cy = int(height * np.random.uniform(0.2, 0.8))
    base_radius = int(min(width, height) * np.random.uniform(0.08, 0.22))

    # 2. יצירת צורת הכתם המרכזי (אורגני ולא-סימטרי)
    num_points = np.random.randint(18, 30)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # שינוי אקראי ברדיוס ליצירת גליות
    radii = base_radius * np.random.uniform(0.5, 1.4, size=num_points)

    pts = []
    for a, r in zip(angles, radii):
        px = int(cx + r * np.cos(a))
        py = int(cy + r * np.sin(a))
        pts.append([px, py])

    stain_poly = np.array(pts, np.int32)

    # 3. יצירת מסיכות עבור מרכז הכתם ועבור טבעת השוליים (Coffee Ring)
    mask_fill = np.zeros((height, width), dtype=np.float32)
    cv2.fillPoly(mask_fill, [stain_poly], 1.0)

    # קו מתאר עבור השוליים הכהים (Coffee Ring Effect)
    mask_ring = np.zeros((height, width), dtype=np.float32)
    ring_thickness = int(base_radius * 0.001)
    cv2.polylines(
        mask_ring,
        [stain_poly],
        isClosed=True,
        color=1.0,
        thickness=ring_thickness,
        lineType=cv2.LINE_AA,
    )

    # 4. הוספת מריחה כיוונית (Smear Effect) - ב-50% מהמקרים
    if np.random.rand() < 0.5:
        smear_angle = np.random.uniform(0, 2 * np.pi)
        smear_len = int(base_radius * np.random.uniform(0.8, 1.8))
        smear_end = (
            int(cx + smear_len * np.cos(smear_angle)),
            int(cy + smear_len * np.sin(smear_angle)),
        )

        # ציור המריחה בעובי דועך
        cv2.line(
            mask_fill,
            (cx, cy),
            smear_end,
            0.8,
            thickness=int(base_radius * 0.6),
            lineType=cv2.LINE_AA,
        )

    # 5. הוספת נתזים וטיפות קטנות מסביב (Splatters)
    num_splatters = np.random.randint(8, 20)
    for _ in range(num_splatters):
        s_dist = base_radius * np.random.uniform(1.1, 2.2)
        s_angle = np.random.uniform(0, 2 * np.pi)
        sx = int(cx + s_dist * np.cos(s_angle))
        sy = int(cy + s_dist * np.sin(s_angle))
        s_size = np.random.randint(1, max(2, int(base_radius * 0.08)))

        if 0 <= sx < width and 0 <= sy < height:
            cv2.circle(mask_fill, (sx, sy), s_size, 0.9, -1)

    # 6. טשטוש רך של המסיכות למעבר הדרגתי
    blur_k = max(5, int(base_radius * 0.25) | 1)
    mask_fill_smooth = cv2.GaussianBlur(mask_fill, (blur_k, blur_k), 0)
    mask_ring_smooth = cv2.GaussianBlur(
        mask_ring, (max(3, blur_k // 2) | 1, max(3, blur_k // 2) | 1), 0
    )

    # 7. הגדרת פלטת צבעים/שינוי ערוצים לפי סוג הכתם
    if is_color:
        if stain_type == "coffee":
            # קפה: הורדה חזקה של כחול, הורדה בינונית של ירוק, מעט אדום (חום-כהה)
            color_factors = [
                1.0 - (0.65 * intensity),  # B
                1.0 - (0.42 * intensity),  # G
                1.0 - (0.22 * intensity),  # R
            ]
        elif stain_type == "tea":
            # תה: גוון צהבהב-כתום קל
            color_factors = [
                1.0 - (0.45 * intensity),
                1.0 - (0.20 * intensity),
                1.0 - (0.05 * intensity),
            ]
        elif stain_type == "white_peel":
            # הבהרה עוצמתית של כל 3 הערוצים כדי שיהפכו ללבנים
            color_factor_val = 1.0 + (1.5 * intensity)
            color_factors = [color_factor_val, color_factor_val, color_factor_val]
        else:  # grease / dark stain
            # כתם שומן/לכלוך כהה אחיד
            factor = 1.0 - (0.50 * intensity)
            color_factors = [factor, factor, factor]

        # החלת הצבע במרכז הכתם
        for c in range(3):
            result[:, :, c] *= 1.0 - (
                (1.0 - color_factors[c]) * mask_fill_smooth
            )

        # החלת ה-Coffee Ring (הכהיה נוספת בשוליים)
        # החלת ה-Coffee Ring (הכהיה נוספת בשוליים)
        ring_darkening = 1.0 - (0.35 * intensity * mask_ring_smooth)
        for c in range(3):
            result[:, :, c] *= ring_darkening

    else:
        # תמונה בשחור-לבן
        darkening = 1.0 - (0.45 * intensity * mask_fill_smooth)
        ring_darkening = 1.0 - (0.30 * intensity * mask_ring_smooth)
        result *= darkening * ring_darkening

    return np.clip(result, 0, 255).astype(np.uint8)

def add_white_mold_abrasion(image, intensity=0.3, skip_probability=0.2):
    """מדמה עובש לבן, שחיקה וקילופי אמולסיה בצורה אורגנית וריאליסטית

    (Coherent / Perlin-like Mold Noise)
    """
    if np.random.rand() < skip_probability:
        return image.copy()

    height, width = image.shape[:2]
    result = image.astype(np.float32)

    # 1. יצירת רעש רציף בתדרים שונים מדמה אשכולות עובש צפופים (Coherent Noise)
    # מייצרים רשת רעש קטנה ומגדילים אותה עם אינטרפולציה רכה
    small_h, small_w = max(10, height // 16), max(10, width // 16)
    low_res_noise1 = np.random.uniform(0, 1, (small_h, small_w)).astype(
        np.float32
    )
    coherent_noise1 = cv2.resize(
        low_res_noise1, (width, height), interpolation=cv2.INTER_CUBIC
    )

    # תדר שני למרקם פנימי עדין יותר
    med_h, med_w = max(20, height // 6), max(20, width // 6)
    low_res_noise2 = np.random.uniform(0, 1, (med_h, med_w)).astype(np.float32)
    coherent_noise2 = cv2.resize(
        low_res_noise2, (width, height), interpolation=cv2.INTER_CUBIC
    )

    # שילוב התדרים
    combined_noise = 0.65 * coherent_noise1 + 0.35 * coherent_noise2

    # 2. הגדרת אזורי הכתמים - חיתוך לפי סף אורגני (Thresholding)
    # ככל ש-intensity גבוה יותר, הכתמים תופסים שטח גדול יותר
    cutoff = 0.82 - (0.28 * intensity)
    mold_mask = np.clip((combined_noise - cutoff) / (1.0 - cutoff), 0.0, 1.0)

    # 3. הוספת מרקם גרגירי צפוף בתוך האזורים (Micro-Structure)
    # יצירת מסיכת חספוס פנימית
    fine_h, fine_w = max(40, height // 2), max(40, width // 2)
    fine_noise = cv2.resize(
        np.random.uniform(0, 1, (fine_h, fine_w)).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    # הכפלת המסיכה האורגנית במרקם הפיזי כדי שהנקודות יהיו צפופות ומחוברות
    detailed_mold_mask = mold_mask * (0.55 + 0.45 * fine_noise)

    # 4. הוספת פסי שחיקה אנכיים דקים (Vertical Abrasion Streaks)
    num_streaks = np.random.randint(5, 20)
    streaks_mask = np.zeros((height, width), dtype=np.float32)

    for _ in range(num_streaks):
        sx = int(width * np.random.uniform(0.05, 0.95))
        sy = int(height * np.random.uniform(0.1, 0.7))
        length = int(height * np.random.uniform(0.1, 0.35))

        ex = sx + np.random.randint(-2, 3)
        ey = min(height - 1, sy + length)

        cv2.line(
            streaks_mask,
            (sx, sy),
            (ex, ey),
            1.0,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # שחיקה אנכית מופיעה בעיקר באזורי העובש והשוליים
    streaks_mask *= cv2.GaussianBlur(mold_mask, (15, 15), 0) * 1.2
    final_mask = np.clip(detailed_mold_mask + streaks_mask, 0.0, 1.0)

    # 5. טשטוש עדין של המסיכה הסופית למעבר טבעי
    final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

    # 6. החלת הצבע הלבן-גרידי (Off-white / Chalky Paper)
    white_chalk = np.array([242, 245, 238], dtype=np.float32)  # BGR

    if len(image.shape) == 3:
        for c in range(3):
            alpha = final_mask * (0.75 + 0.25 * intensity)
            result[:, :, c] = (result[:, :, c] * (1.0 - alpha)) + (
                white_chalk[c] * alpha
            )
    else:
        alpha = final_mask * (0.75 + 0.25 * intensity)
        result = (result * (1.0 - alpha)) + (242.0 * alpha)

    return np.clip(result, 0, 255).astype(np.uint8)







def apply_manipulations(image, operations):
    """
    Apply a sequence of image manipulations with optional custom parameters.

    Parameters
    ----------
    image : np.ndarray
        The input image.

    operations : list of str or tuple
        A list of operations. Each item can be:
        - "blur" -> uses default parameters
        - ("blur", 5) -> passes positional parameter (sigma=5)
        - ("contrast", {"alpha": 0.8, "beta": 20}) -> passes keyword arguments

    Returns
    -------
    np.ndarray
        The manipulated image.
    """

    manipulation_functions = {
        "blur": blur,
        "film grain": film_grain,
        "sepia": sepia,
        "desaturation": desaturation,
        "vignette": vignette,
        "contrast": change_contrast,
        "gamma": gamma_correction,
        "cracks": cracks,
        "add_dust_and_flecks":add_dust_and_flecks,
        "add_stain":add_stain,
        "add_blur_stain":add_blur_stain,
        "add_random_heavy_tear":add_random_heavy_tear,
        "add_spilled_stain": add_spilled_stain,
        "add_white_mold_abrasion": add_white_mold_abrasion
    }

    for item in operations:
        args = []
        kwargs = {}

        if isinstance(item, str):
            op_name = item

        elif isinstance(item, (tuple, list)):
            op_name = item[0]
            param = item[1]

            if isinstance(param, dict):
                kwargs = param
            else:
                args = [param]
        else:
            raise ValueError(f"Invalid operation format: {item}")

        op_key = op_name.lower().strip()

        if op_key not in manipulation_functions:
            raise ValueError(f"Unknown manipulation: '{op_name}'")

        func = manipulation_functions[op_key]
        image = func(image, *args, **kwargs)

    return image

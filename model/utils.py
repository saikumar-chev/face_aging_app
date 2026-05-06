import cv2
import numpy as np

def apply_aging_effect(pil_image, effect_type, age=50):
    """
    Apply aging or de-aging effect to a face image, simulating realistic facial changes.
    
    Args:
        pil_image: PIL Image object (RGB)
        effect_type: String, either "Age (👵)" or "De-Age (👶)"
        age: Integer (1-100) representing target age
    
    Returns:
        Processed image as numpy array (RGB)
    """
    # Convert PIL image to OpenCV format (BGR)
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Normalize age to a 0-1 scale for effect intensity
    normalized_age = max(1, min(age, 100)) / 100.0
    strength = normalized_age * 10  # Scale to 0-10 for effect control

    if effect_type == "Age (👵)":
        # Aging effect: wrinkles, age spots, sagging skin, gray hair
        # 1. Simulate wrinkles using edge detection and morphological operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        wrinkles = cv2.dilate(edges, kernel, iterations=int(strength / 2))
        wrinkles = cv2.GaussianBlur(wrinkles, (5, 5), 0)
        aged = cv2.addWeighted(img, 1.0, cv2.cvtColor(wrinkles, cv2.COLOR_GRAY2BGR), 0.1 * strength, 0.0)
        
        # 2. Add age spots (random small patches of darker color)
        hsv = cv2.cvtColor(aged, cv2.COLOR_BGR2HSV).astype(np.float32)
        age_spots = np.zeros_like(aged, dtype=np.uint8)
        num_spots = int(strength * 20)
        for _ in range(num_spots):
            x, y = np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])
            radius = np.random.randint(2, 5)
            cv2.circle(age_spots, (x, y), radius, (int(20 * strength), 50, 50), -1)
        aged = cv2.addWeighted(aged, 1.0, age_spots, 0.3, 0.0)
        
        # 3. Dull skin tone and reduce brightness
        hsv[:, :, 1] *= (1.0 - normalized_age * 0.3)  # Reduce saturation
        hsv[:, :, 2] *= (1.0 - normalized_age * 0.4)  # Reduce brightness
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        aged = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 4. Simulate gray hair in high-brightness areas
        gray_mask = cv2.cvtColor(aged, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY)
        gray_hair = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
        aged = np.where(mask[:, :, None] == 255, gray_hair, aged)
        
        result = aged

    elif effect_type == "De-Age (👶)":
        # De-aging effect: smooth skin, fuller contours, vibrant colors
        # 1. Smooth skin using edge-preserving filter
        deaged = cv2.bilateralFilter(img, d=7 + int(strength * 3), sigmaColor=60 + strength * 10, sigmaSpace=60)
        
        # 2. Simulate fuller cheeks by subtle Gaussian blur on face center
        mask = np.zeros_like(deaged, dtype=np.uint8)
        height, width = deaged.shape[:2]
        cv2.ellipse(mask, (width//2, height//2), (int(width * 0.2), int(height * 0.3)), 0, 0, 360, (255, 255, 255), -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        deaged = cv2.addWeighted(deaged, 1.0, cv2.GaussianBlur(deaged, (15, 15), 0), 0.2 * strength, 0.0)
        
        # 3. Brighten and enhance color vibrancy
        hsv = cv2.cvtColor(deaged, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= (1.0 + normalized_age * 0.5)  # Increase saturation
        hsv[:, :, 2] *= (1.0 + normalized_age * 0.4)  # Increase brightness
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        deaged = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 4. Enhance hair color (increase vibrancy in high-brightness areas)
        gray_mask = cv2.cvtColor(deaged, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_mask, 150, 255, cv2.THRESH_BINARY)
        vibrant_hair = cv2.cvtColor(deaged, cv2.COLOR_BGR2HSV).astype(np.float32)
        vibrant_hair[:, :, 1] *= 1.2  # Boost hair color saturation
        vibrant_hair = np.clip(vibrant_hair, 0, 255).astype(np.uint8)
        vibrant_hair = cv2.cvtColor(vibrant_hair, cv2.COLOR_HSV2BGR)
        deaged = np.where(mask[:, :, None] == 255, vibrant_hair, deaged)
        
        result = deaged

    else:
        result = img

    # Convert back to RGB for PIL compatibility
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result
import io
import numpy as np
from PIL import Image


def create_gradcam_overlay(image_source, heatmap, output_path, alpha=0.4):

    # Open original MRI image
    image = Image.open(
        io.BytesIO(image_source)
        if isinstance(image_source, (bytes, bytearray))
        else image_source
    ).convert("RGB")

    # Resize heatmap to original image dimensions
    heatmap_img = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    )

    heatmap_img = heatmap_img.resize(
        image.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.asarray(heatmap_img, dtype=np.float32) / 255.0

    # Apply JET-like color mapping
    red = np.clip(1.5 - np.abs(4 * heatmap_array - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * heatmap_array - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * heatmap_array - 1), 0, 1)

    colored_heatmap = np.stack(
        [red, green, blue],
        axis=-1
    )

    original = np.asarray(image, dtype=np.float32) / 255.0

    # Blend original MRI and heatmap
    overlay = (
        (1 - alpha) * original +
        alpha * colored_heatmap
    )

    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    # Save visualization
    result = Image.fromarray(overlay)
    result.save(output_path)

    return output_path
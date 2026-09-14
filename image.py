from PIL import Image, ImageDraw


def create_title_slide_bg(input_path, output_path, expand_pixels=800):
    # Load the original tunnel image
    img = Image.open(input_path).convert("RGBA")
    orig_w, orig_h = img.size

    # 1. Create a new, wider white canvas for the presentation slide
    new_w = orig_w + expand_pixels
    final_canvas = Image.new("RGBA", (new_w, orig_h), (255, 255, 255, 255))

    # 2. Paste the original tunnel image on the left side
    final_canvas.paste(img, (0, 0))

    # 3. Create a grayscale mask to control the white fade
    mask = Image.new("L", (new_w, orig_h), 0)
    draw = ImageDraw.Draw(mask)

    fade_start = int(orig_w * 0.55)  # Start fading 55% across the image
    fade_end = orig_w  # Fully white by the original right edge

    # Apply the horizontal gradient fade
    for x in range(fade_start, fade_end):
        opacity = int(255 * (x - fade_start) / (fade_end - fade_start))
        draw.line([(x, 0), (x, orig_h)], fill=opacity)

    # Keep the newly expanded right section solid white
    draw.rectangle([(fade_end, 0), (new_w, orig_h)], fill=255)

    # ---------------------------------------------------------
    # CORNER-FOCUSED GRADIENT TWEAK
    # ---------------------------------------------------------
    # Expand vertical reach to 75% for a smoother blend
    for y in range(0, int(orig_h * 0.60)):
        for x in range(0, new_w):
            current_val = mask.getpixel((x, y))
            
            # Type guard to resolve the Pylance warning
            assert isinstance(current_val, int)

            # Normalize distances: 1.0 at the absolute top-right, fading down to 0.0
            x_factor = x / new_w
            y_factor = 1 - (y / (orig_h * 0.75))
            
            # Concentrate the fade purely in the corner using an exponent (1.5),
            # and scale it up heavily (400) to intensify the "white-out" effect
            corner_intensity = (x_factor * y_factor) ** 1.5
            bias = int(400 * corner_intensity)
            
            mask.putpixel((x, y), min(255, current_val + bias))

    # 4. Overlay pure white onto the canvas using the gradient mask
    white_overlay = Image.new("RGBA", (new_w, orig_h), (255, 255, 255, 255))
    final_bg = Image.composite(white_overlay, final_canvas, mask)

    # Save the final presentation background
    final_bg.convert("RGB").save(output_path)
    print(f"[*] Expanded presentation background saved to {output_path}")


# Run the function
create_title_slide_bg("pic2.png", "title_slide_bg.png")
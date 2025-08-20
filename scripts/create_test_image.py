#!/usr/bin/env python
"""
Create a simple test document image for TriadFuse testing
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_test_document(output_path: str = "test_document.png"):
    """Create a simple invoice-like test document"""
    # Create white background
    width, height = 600, 800
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a basic font, fall back to default if not available
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        # Use default font if system fonts not available
        font_large = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw header
    draw.text((50, 50), "INVOICE", font=font_large, fill="black")
    draw.text((50, 90), "Invoice #: INV-2024-001", font=font_normal, fill="black")
    draw.text((50, 120), "Date: 2024-01-15", font=font_normal, fill="black")

    # Draw company info
    draw.rectangle((50, 160, 550, 250), outline="gray", width=1)
    draw.text((60, 170), "ACME Corporation", font=font_normal, fill="black")
    draw.text((60, 195), "123 Business Street", font=font_small, fill="black")
    draw.text((60, 215), "City, State 12345", font=font_small, fill="black")

    # Draw table header
    y_pos = 290
    draw.line((50, y_pos, 550, y_pos), fill="black", width=2)
    draw.text((60, y_pos + 10), "Item", font=font_normal, fill="black")
    draw.text((250, y_pos + 10), "Quantity", font=font_normal, fill="black")
    draw.text((350, y_pos + 10), "Price", font=font_normal, fill="black")
    draw.text((450, y_pos + 10), "Total", font=font_normal, fill="black")
    draw.line((50, y_pos + 40, 550, y_pos + 40), fill="black", width=1)

    # Draw items
    items = [
        ("Product A", "2", "$50.00", "$100.00"),
        ("Product B", "1", "$75.00", "$75.00"),
        ("Service C", "3", "$25.00", "$75.00"),
    ]

    y_offset = y_pos + 50
    for item, qty, price, total in items:
        draw.text((60, y_offset), item, font=font_small, fill="black")
        draw.text((260, y_offset), qty, font=font_small, fill="black")
        draw.text((350, y_offset), price, font=font_small, fill="black")
        draw.text((450, y_offset), total, font=font_small, fill="black")
        y_offset += 30

    # Draw total
    draw.line((350, y_offset + 10, 550, y_offset + 10), fill="black", width=2)
    draw.text((350, y_offset + 20), "Subtotal:", font=font_normal, fill="black")
    draw.text((450, y_offset + 20), "$250.00", font=font_normal, fill="black")
    draw.text((350, y_offset + 45), "Tax (8%):", font=font_normal, fill="black")
    draw.text((450, y_offset + 45), "$20.00", font=font_normal, fill="black")
    draw.line((350, y_offset + 70, 550, y_offset + 70), fill="black", width=1)
    draw.text((350, y_offset + 80), "TOTAL:", font=font_large, fill="black")
    draw.text((450, y_offset + 80), "$270.00", font=font_large, fill="black")

    # Add footer
    draw.text((50, height - 100), "Thank you for your business!", font=font_normal, fill="gray")
    draw.text((50, height - 70), "Payment due within 30 days", font=font_small, fill="gray")

    # Save image
    img.save(output_path)
    print(f"Test document saved to: {output_path}")


if __name__ == "__main__":
    import sys

    output = sys.argv[1] if len(sys.argv) > 1 else "test_document.png"
    create_test_document(output)

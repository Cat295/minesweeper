import pyautogui

# Define the region (x, y, width, height)
region = (889, 509, 995-889, 635-509)

# Take a screenshot of the defined region
screenshot = pyautogui.screenshot(region=region)

# Save the screenshot to a file
screenshot.save('partial_screenshot.png')  # Save as PNG file

print("Screenshot saved as 'partial_screenshot.png'.")

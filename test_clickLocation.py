import pyautogui
import tkinter as tk

# Function to update and display the mouse position
def show_mouse_position(event):
    # Get the current mouse position
    x, y = pyautogui.position()
    position_label.config(text=f"Mouse position: X = {x}, Y = {y}")

# Create a tkinter window
window = tk.Tk()
window.title("Mouse Position Tracker")

# Create a label to display the mouse position
position_label = tk.Label(window, text="Click to show mouse position", font=('Helvetica', 16))
position_label.pack(pady=20)

# Bind the mouse click event to show the mouse position
window.bind("<Button-1>", show_mouse_position)

# Start the tkinter loop
window.mainloop()

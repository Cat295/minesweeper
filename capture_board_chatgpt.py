import pyautogui
import pytesseract
import cv2
import numpy as np
import time

# Screen coordinates for the Minesweeper board (set them according to your screen)
BOARD_TOP_LEFT = (1118, 588)  # Set the top-left corner of the board
CELL_SIZE = 36  # Adjust this based on the size of the cells in pixels
BOARD_SIZE = 9
NUMBER_OF_MINE = 10

# Load reference images
list_imgs = ['sample/mine.png', 'sample/unveiled_1.png', 'sample/unveiled_2.png', 'sample/unveiled_3.png', 'sample/unveiled_4.png', 'sample/unveiled_5.png','sample/unveiled_5_2.png', 'sample/unveiled_6.png', 'sample/unveiled_7.png', 'sample/unveiled_8.png']
unreveiled_list = [cv2.imread(cell_img, cv2.IMREAD_COLOR) for cell_img in list_imgs]

imgs_numbers_list = ['sample/1.png', 'sample/2.png', 'sample/3.png', 'sample/4.png']
numbers_list = [cv2.imread(cell_img, cv2.IMREAD_COLOR) for cell_img in imgs_numbers_list]

imgs_reveiled_list = ['sample/revealed_1.png', 'sample/revealed_2.png', 'sample/revealed_3.png', 'sample/revealed_4.png', 'sample/revealed_5.png', 'sample/revealed_6.png']
reveiled_list = [cv2.imread(cell_img, cv2.IMREAD_COLOR) for cell_img in imgs_reveiled_list]

def capture_board():
    """Captures the entire game board from the screen"""
    board_width = CELL_SIZE * BOARD_SIZE
    board_height = CELL_SIZE * BOARD_SIZE
    
    # Capture the entire board at once
    screenshot = pyautogui.screenshot(region=(BOARD_TOP_LEFT[0], BOARD_TOP_LEFT[1], board_width, board_height))
    
    # Convert the screenshot to a format usable by OpenCV
    open_cv_image = np.array(screenshot)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    return open_cv_image

def classify_cell(cell_image):
    """Classifies a given cell based on predefined images"""
    cell_state = None
    threshold_hit = 0.8  # Similarity threshold for cell to hit

    max_val = 0  # Highest similarity found
    cell_class = None

    # Compare against unreveiled cells (e.g., mines)
    for i, cell_img in enumerate(unreveiled_list):
        result = cv2.matchTemplate(cell_image, cell_img, cv2.TM_CCOEFF_NORMED)
        _, max_val_local, _, _ = cv2.minMaxLoc(result)
        if max_val_local > max_val:
            max_val = max_val_local
            cell_class = f"Unrevealed: {list_imgs[i]}"
        if max_val_local > 0.7:
            return CELL_COVERED

    # Compare against numbers
    for i, cell_img in enumerate(numbers_list):
        result = cv2.matchTemplate(cell_image, cell_img, cv2.TM_CCOEFF_NORMED)
        _, max_val_local, _, _ = cv2.minMaxLoc(result)
        if max_val_local > max_val:
            max_val = max_val_local
            cell_class = f"Number: {i + 1}"
        if max_val_local > threshold_hit:
            return i + 1

    # Compare against revealed cells
    for i, cell_img in enumerate(reveiled_list):
        result = cv2.matchTemplate(cell_image, cell_img, cv2.TM_CCOEFF_NORMED)
        _, max_val_local, _, _ = cv2.minMaxLoc(result)
        if max_val_local > max_val:
            max_val = max_val_local
            cell_class = f"Revealed: {imgs_reveiled_list[i]}"
        if max_val_local > 0.55:
            return 0

    if cell_class:
        print(f"Detected {cell_class} with max hit = {round(max_val, 2)}")
    
    return cell_state if cell_state is not None else "?"

def convert_board_capture_to_matrix(board_image):
    """Extracts an individual cell image from the board and classifies them, returns a NumPy array."""
    board_matrix = []
    
    for row in range(BOARD_SIZE):
        current_row = []
        for col in range(BOARD_SIZE):
            # Extract the region for the current cell
            x_start = col * CELL_SIZE
            y_start = row * CELL_SIZE
            cell_image = board_image[y_start:y_start + CELL_SIZE, x_start:x_start + CELL_SIZE]
            
            # Classify the cell
            cell_state = classify_cell(cell_image)
            
            current_row.append(cell_state)
        board_matrix.append(current_row)
    
    # Convert to NumPy array
    board_matrix_np = np.array(board_matrix)
    
    return board_matrix_np


def show_board_as_matrix(board_matrix):
    """Displays the board matrix in a human-readable format"""
    for row in board_matrix:
        max_width = 1  # max_width = 3
        print(' '.join(f"{str(x).rjust(max_width)}" for x in row))

def main():
    while True:
        # Capture the board
        board_image = capture_board()

        # Convert the board capture to a matrix (NumPy array)
        board_matrix_np = convert_board_capture_to_matrix(board_image)

        # Show the board as a NumPy matrix
        print(board_matrix_np)
        
        # Add a delay to avoid capturing too frequently
        time.sleep(1)  # Adjust as needed
        quit()

if __name__ == "__main__":
    main()


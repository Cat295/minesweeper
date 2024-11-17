import sys
import pyautogui
import pytesseract
import cv2
import numpy as np
import time
import os
import glob

# Screen coordinates for the Minesweeper board (set them according to your screen)
BOARD_TOP_LEFT = (920, 463)  # Set the top-left corner of the board
CELL_SIZE = 36  # Adjust this based on the size of the cells in pixels
BOARD_SIZE_COL = 20
BOARD_SIZE_ROW = 16
NUMBER_OF_MINE = 10

screen_width, screen_height = pyautogui.size() 
if (screen_width, screen_height) == (2560, 1440): 
    print("2K") 
    OFFSET_X_LVL_1 = 1118
    OFFSET_Y_LVL_1 = 588
    OFFSET_X_LVL_2 = 1064
    OFFSET_Y_LVL_2 = 552
    OFFSET_X_LVL_3 = 1010
    OFFSET_Y_LVL_3 = 516
    OFFSET_X_LVL_4 = 956
    OFFSET_Y_LVL_4 = 498
    OFFSET_X_LVL_5 = 920
    OFFSET_Y_LVL_5 = 463
else:
    print("FULLHD") 
    OFFSET_X_LVL_1 = 1118 - 320  # 798
    OFFSET_Y_LVL_1 = 588 - 180   # 408
    OFFSET_X_LVL_2 = 1064 - 320  # 744
    OFFSET_Y_LVL_2 = 552 - 180   # 372
    OFFSET_X_LVL_3 = 1010 - 320  # 690
    OFFSET_Y_LVL_3 = 516 - 180   # 336
    OFFSET_X_LVL_4 = 956 - 320   # 636
    OFFSET_Y_LVL_4 = 498 - 180   # 318
    OFFSET_X_LVL_5 = 920 - 320   # 600
    OFFSET_Y_LVL_5 = 463 - 180   # 283

# Load reference images

directory_unrevealed = './sample/unrevealed'
# Get all PNG files in the directory_unrevealed
list_unrevealed_imgs = glob.glob(os.path.join(directory_unrevealed, '*.png'))
unrevealed_list = [
    cv2.imread(cell_img, cv2.IMREAD_COLOR) 
    for cell_img in list_unrevealed_imgs
]

imgs_numbers_list = [
    'sample/1.png', 
    'sample/2.png', 
    'sample/3.png', 
    'sample/4.png', 
    'sample/5.png',
    'sample/6.png',
    'sample/5_2.png'
]
numbers_list = [
    cv2.imread(cell_img, cv2.IMREAD_COLOR) 
    for cell_img in imgs_numbers_list
]

directory_revealed = './sample/revealed'
revealed_list_imgs = glob.glob(os.path.join(directory_revealed, '*.png'))
revealed_list = [
    cv2.imread(cell_img, cv2.IMREAD_COLOR) 
    for cell_img in revealed_list_imgs
]

directory_mine = './sample/mine'
imgs_mine_list = glob.glob(os.path.join(directory_mine, '*.png'))
mine_list = [
    cv2.imread(cell_img, cv2.IMREAD_COLOR) 
    for cell_img in imgs_mine_list
]

directory_item = './sample/item'
item_list_imgs = glob.glob(os.path.join(directory_item, '*.png'))
item_list = [
    cv2.imread(cell_img, cv2.IMREAD_COLOR) 
    for cell_img in item_list_imgs
]

def capture_board():
    """Captures the game board from the screen"""
    # Calculate the bottom-right corner of the board
    board_width = CELL_SIZE * BOARD_SIZE_COL
    board_height = CELL_SIZE * BOARD_SIZE_ROW
    board_bottom_right = (BOARD_TOP_LEFT[0] + board_width, BOARD_TOP_LEFT[1] + board_height)
    
    # Capture the region of the screen containing the board
    screenshot = pyautogui.screenshot(region=(BOARD_TOP_LEFT[0], BOARD_TOP_LEFT[1], board_width, board_height))
    ### screenshot.save('sample/partial_screenshot.png')  # Save as PNG file
    # Convert the screenshot to a format usable by OpenCV
    open_cv_image = np.array(screenshot)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

    
    return open_cv_image

################
def classify_cell(board_screenshot, row, col):
    # Classify the cell by comparing it with reference images
    cell_state = None
    threshold_hit = 0.8  # Similarity threshold for cell to hit

    maxhit = 0

    for i, cell_img in enumerate(unrevealed_list):
        result_hit = cv2.matchTemplate(board_screenshot, cell_img, cv2.TM_CCOEFF_NORMED)
        if result_hit > threshold_hit:
            print(f"Detected unrevealed {row}_{col}: {list_unrevealed_imgs[i]}")
            cell_state = '#'
        if result_hit > maxhit:
            maxhit = result_hit
    if cell_state == None:
        for i, cell_img in enumerate(numbers_list):
            result_hit = cv2.matchTemplate(board_screenshot, cell_img, cv2.TM_CCOEFF_NORMED)
            if result_hit > threshold_hit:
                print(f"Detected number {row}_{col}: {imgs_numbers_list[i]}")
                cell_state = i+1
                break
            if result_hit > maxhit:
                maxhit = result_hit
    if cell_state == None:
        for i, cell_img in enumerate(revealed_list):
            result_hit = cv2.matchTemplate(board_screenshot, cell_img, cv2.TM_CCOEFF_NORMED)
            if result_hit > threshold_hit-0.15:
                print(f"Detected revealed {row}_{col}: {revealed_list_imgs[i]}")
                cell_state = '-'
                break
            if result_hit > maxhit:
                maxhit = result_hit
    if cell_state == None:
        for i, cell_img in enumerate(mine_list):
            result_hit = cv2.matchTemplate(board_screenshot, cell_img, cv2.TM_CCOEFF_NORMED)
            if result_hit > threshold_hit:
                print(f"Detected mine {row}_{col}: {imgs_mine_list[i]}")
                cell_state = '*'
                break
            if result_hit > maxhit:
                maxhit = result_hit
    
    for i, cell_img in enumerate(item_list):
        result_hit = cv2.matchTemplate(board_screenshot, cell_img, cv2.TM_CCOEFF_NORMED)
        if result_hit > threshold_hit:
            print(f"Detected item {row}_{col}: {item_list_imgs[i]}")
            cell_state = '$'
            break
        if result_hit > maxhit:
            maxhit = result_hit

    if cell_state == None:
        print(f"Max hit = {maxhit}")
        cell_state = '?'
    
    return cell_state
###################

def convert_board_capture_to_matrix(board_image):
    """Extracts an individual cell image from the board and classifies them."""
    board_matrix = []
    
    for row in range(BOARD_SIZE_ROW):
        current_row = []
        for col in range(BOARD_SIZE_COL):
            # Extract the region for the current cell
            screenshot = pyautogui.screenshot(region=(BOARD_TOP_LEFT[0] + col * CELL_SIZE, BOARD_TOP_LEFT[1] + row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            screenshot.save('./img/' + str(row+1) + '_' + str(col+1) + '.png')  # Save as PNG file

            # Convert the screenshot to a format that OpenCV can work with
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            cell_state = classify_cell(screenshot, row, col)

            current_row.append(cell_state)
        board_matrix.append(current_row)
    
    return board_matrix

def show_board_as_matrix(board_matrix):
    """Displays the board matrix in a human-readable format"""
    for row in board_matrix:
        # print(' '.join(map(str, row)))
        max_width = 2  # max_width = 3
        print(' '.join(f"{str(x).rjust(max_width)}" for x in row))

def main(level):
    global BOARD_TOP_LEFT, BOARD_SIZE_COL, BOARD_SIZE_ROW, NUMBER_OF_MINE
    if level == 1:
        BOARD_TOP_LEFT = (OFFSET_X_LVL_1, OFFSET_Y_LVL_1)
        BOARD_SIZE_COL = 9
        BOARD_SIZE_ROW = 9
        NUMBER_OF_MINE = 10
    elif level == 2:
        BOARD_TOP_LEFT = (OFFSET_X_LVL_2, OFFSET_Y_LVL_2)
        BOARD_SIZE_COL = 12
        BOARD_SIZE_ROW = 11
        NUMBER_OF_MINE = 19
    elif level == 3:
        BOARD_TOP_LEFT = (OFFSET_X_LVL_3, OFFSET_Y_LVL_3)
        BOARD_SIZE_COL = 15
        BOARD_SIZE_ROW = 13
        NUMBER_OF_MINE = 32
    elif level == 4:
        BOARD_TOP_LEFT = (OFFSET_X_LVL_4, OFFSET_Y_LVL_4)
        BOARD_SIZE_COL = 18
        BOARD_SIZE_ROW = 14
        NUMBER_OF_MINE = 47
    else:
        BOARD_TOP_LEFT = (OFFSET_X_LVL_5, OFFSET_Y_LVL_5)
        BOARD_SIZE_COL = 20
        BOARD_SIZE_ROW = 16
        NUMBER_OF_MINE = 66
    while True:
        # Capture the board
        board_image = capture_board()

        # Convert the board capture to a matrix
        board_matrix = convert_board_capture_to_matrix(board_image)

        # Show the board as a matrix
        show_board_as_matrix(board_matrix)
        board_matrix_np = np.array(board_matrix)

        if np.any(board_matrix_np == '?') == False:
            quit()
        # Add a delay so that we're not capturing too frequently
        # time.sleep(1)
        quit()

if __name__ == "__main__":
    # print(sys.argv)
    main(int(sys.argv[1]))

''' Class for simulating n-dimensional minesweeper games:
generate board, input a move, answer with resulting boards etc.
(Visualization only works for n in (2, 3, 4))
'''

import random

from dataclasses import dataclass
import numpy as np

from PIL import Image, ImageDraw


import pyautogui
import pytesseract
import cv2
# import numpy as np
import time
import os
import glob

# Global settings for Minesweeper
BOARD_TOP_LEFT = (1118, 588)  # Default top-left corner for LEVEL 1
CELL_SIZE = 36  # Adjust this based on the size of the cells in pixels
BOARD_SIZE_COL = 9  # Default board size for LEVEL 1
BOARD_SIZE_ROW = 9  # Default board size for LEVEL 1
NUMBER_OF_MINE = 10  # Default number of mines for LEVEL 1

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



def printBoardInfo():
    print(f'BOARD_TOP_LEFT: {BOARD_TOP_LEFT}')
    print(f'BOARD_SIZE_COL: {BOARD_SIZE_COL}')
    print(f'BOARD_SIZE_ROW: {BOARD_SIZE_ROW}')

@dataclass
class GameSettings:
    ''' Class to hold game settings:
    dimensions of the board, number of mines, use wrap around
    '''
    shape: tuple = (8, 8)
    mines: int = 10
    density: float = 0
    wrap_around: bool = False

    def __post_init__(self):
        # Calculate mine density (used only for information)
        self.density = self.mines / np.prod(self.shape)

    def __str__(self):
        output = ""
        output += f"Dims:{len(self.shape)}, "
        output += f"Shape:{self.shape}, "
        output += f"Volume:{np.prod(self.shape)}, "
        output += f"Mines:{self.mines}, "
        output += f"Density:{self.density:.1%}, "
        output += f"Wrap:{'yes' if self.wrap_around else 'no'}, "
        return output


# Presets for main game sizes
# Classical minesweeper difficulties
GAME_DOTA_LVL_1 = GameSettings((9, 9), 10)
GAME_DOTA_LVL_2 = GameSettings((12, 11), 19)
GAME_DOTA_LVL_3 = GameSettings((15, 13), 32)
GAME_DOTA_LVL_4 = GameSettings((18, 14), 47)
GAME_DOTA_LVL_5 = GameSettings((20, 16), 66)

# Cell types
CELL_MINE = -1
# Others are for the self.uncovered field
# Never clicked
CELL_COVERED = -2
# Marked mine, but actually isn't
CELL_FALSE_MINE = -3
# Explosion (clicked safe, but it was a mine)
CELL_EXPLODED_MINE = -4

# Characters to show for different statuses
# 80 is the highest possible number of neighbors (in a 4d game)
LEGEND = {**{
    CELL_MINE: "*",
    CELL_COVERED: " ",
    CELL_FALSE_MINE: "X",
    CELL_EXPLODED_MINE: "!",
    0: "."
}, **{i: str(i) for i in range(1, 81)}}

# MinesweeperGame.status: returned by the do_move, tells the result of the move
STATUS_ALIVE = 0
STATUS_DEAD = 1
STATUS_WON = 2
STATUS_MESSAGES = {STATUS_ALIVE: "Still alive",
                   STATUS_DEAD: "You died",
                   STATUS_WON: "You won"}


##############################################
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
    'sample/5_2.png',
    'sample/1_2.png',
    'sample/7.png'
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

screen_width, screen_height = pyautogui.size()
half_width = screen_width // 2
half_height = screen_height // 2

imgs_dead_list = [
    'sample/dead_1.png', 
    'sample/dead_2.png', 
    'sample/dead_3.png'
]
dead_list = [
    cv2.imread(cell_img, cv2.IMREAD_COLOR) 
    for cell_img in imgs_dead_list
]

won_template = cv2.imread('./sample/won.png', cv2.IMREAD_COLOR)

def check_template_match(screenshot, template_list, threshold=0.8):
    """ Helper function to check if any template in template_list matches with a threshold """
    for i, template in enumerate(template_list):
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > threshold:
            return i, max_val
    return None, 0

def screenshot_board():
    """Captures the entire game board from the screen"""
    board_width = CELL_SIZE * BOARD_SIZE_COL
    board_height = CELL_SIZE * BOARD_SIZE_ROW
    
    # Capture the entire board at once
    screenshot = pyautogui.screenshot(region=(BOARD_TOP_LEFT[0], BOARD_TOP_LEFT[1], board_width, board_height))
    screenshot.save('ALL_BOARD.png')
    # Convert the screenshot to a format usable by OpenCV
    open_cv_image = np.array(screenshot)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    return open_cv_image

def is_game_end():
    # Capture a screenshot of the relevant region
    screen_width, screen_height = pyautogui.size() 
    if (screen_width, screen_height) == (2560, 1440):
        screenshot_check = pyautogui.screenshot(region=(320, 180, 1920, 1080))
    else:
        screenshot_check = pyautogui.screenshot()
    screenshot_check = cv2.cvtColor(np.array(screenshot_check), cv2.COLOR_RGB2BGR)

    # Check for dead cells
    for cell_img in dead_list:
        result = cv2.matchTemplate(screenshot_check, cell_img, cv2.TM_CCOEFF_NORMED)
        _, max_val_dead, _, _ = cv2.minMaxLoc(result)
        if max_val_dead > 0.8:
            print('Dead detected')
            quit()

    # Check for no more moves (unrevealed cells)
    for cell_img in unrevealed_list:
        result = cv2.matchTemplate(screenshot_check, cell_img, cv2.TM_CCOEFF_NORMED)
        _, max_val_unrevealed, _, _ = cv2.minMaxLoc(result)
        if max_val_unrevealed < 0.4:
            print('No more move detected')
            quit()

    # Check for win
    result = cv2.matchTemplate(screenshot_check, won_template, cv2.TM_CCOEFF_NORMED)
    _, max_val_won, _, _ = cv2.minMaxLoc(result)
    if max_val_won > 0.8:
        print('Won detected')
        quit()

    return False


def switch_case(value):
    #img_order: actual_num
    switch = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
        6: 5,
        7: 1,
        8: 7
    }
    return switch.get(value, 0)

def classify_cell(cell_image, row, col):
    """Classifies a given cell based on predefined images"""
    cell_state = None
    threshold_hit = 0.8  # Similarity threshold for cell to hit

    max_val = 0  # Highest similarity found
    cell_class = None

    # Compare against unreveiled cells (e.g., mines)
    for i, cell_img in enumerate(unrevealed_list):
        result = cv2.matchTemplate(cell_image, cell_img, cv2.TM_CCOEFF_NORMED)
        _, max_val_local, _, _ = cv2.minMaxLoc(result)
        if max_val_local > max_val:
            max_val = max_val_local
            cell_class = f"Unrevealed: {list_unrevealed_imgs[i]}"
        if max_val_local >= threshold_hit:
            return CELL_COVERED

    # Compare against numbers
    for i, cell_img in enumerate(numbers_list):
        result = cv2.matchTemplate(cell_image, cell_img, cv2.TM_CCOEFF_NORMED)
        _, max_val_local, _, _ = cv2.minMaxLoc(result)
        if max_val_local > max_val:
            max_val = max_val_local
            cell_class = f"Number: {switch_case(i)}"
        if max_val_local >= threshold_hit:
            #return number bases on img order
            return switch_case(i)

    # Compare against revealed cells
    for i, cell_img in enumerate(revealed_list):
        result = cv2.matchTemplate(cell_image, cell_img, cv2.TM_CCOEFF_NORMED)
        _, max_val_local, _, _ = cv2.minMaxLoc(result)
        if max_val_local > max_val:
            max_val = max_val_local
            cell_class = f"Revealed: {revealed_list_imgs[i]}"
        if max_val_local >= threshold_hit-0.2:
            return 0
    
    # Compare against Mine cells
    for i, cell_img in enumerate(mine_list):
        result = cv2.matchTemplate(cell_image, cell_img, cv2.TM_CCOEFF_NORMED)
        _, max_val_local, _, _ = cv2.minMaxLoc(result)
        if max_val_local > max_val:
            max_val = max_val_local
            cell_class = f"Mine: {imgs_mine_list[i]}"
        if max_val_local >= threshold_hit:
            return CELL_MINE
    
    # ITEM
    for i, cell_img in enumerate(item_list):
        result = cv2.matchTemplate(cell_image, cell_img, cv2.TM_CCOEFF_NORMED)
        _, max_val_local, _, _ = cv2.minMaxLoc(result)
        if max_val_local > max_val:
            max_val = max_val_local
            cell_class = f"Item: {item_list_imgs[i]}"
        if max_val_local >= threshold_hit:
            return "$"

    if cell_class:
        print(f"Detected {cell_class} with max hit = {round(max_val, 2)} Location: {row}_{col}")
    
    return cell_state if cell_state is not None else "?"

item_position_list = []

def convert_board_capture_to_matrix(board_image):
    global item_position_list
    """Extracts an individual cell image from the board and classifies them, returns a NumPy array."""
    board_matrix = []
    item_position_list = []
    count = 0
    for row in range(BOARD_SIZE_ROW):
        current_row = []
        for col in range(BOARD_SIZE_COL):
            # Extract the region for the current cell
            x_start = col * CELL_SIZE
            y_start = row * CELL_SIZE
            cell_image = board_image[y_start:y_start + CELL_SIZE, x_start:x_start + CELL_SIZE]
            
            # Classify the cell
            cell_state = classify_cell(cell_image, row, col)
            while cell_state == '?':
                # unknownCellScreenshot = pyautogui.screenshot(region=(BOARD_TOP_LEFT[0] + col * CELL_SIZE, BOARD_TOP_LEFT[1] + row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                # unknownCellScreenshot.save('./unknown/' + str(row+1) + '_' + str(col+1) + '.png')
                # quit()
                # time.sleep(0.1)
                board_image = screenshot_board()
                cell_image = board_image[y_start:y_start + CELL_SIZE, x_start:x_start + CELL_SIZE]
                cell_state = classify_cell(cell_image, row, col)
                count = count+1
                if count>30:
                    quit()
                # return np.array(board_matrix)
            if cell_state == '$':
                print(f'Detected ITEM at {col} {row}')
                cell_state = CELL_COVERED
                item_position_list.append((col, row))
            current_row.append(cell_state)
        board_matrix.append(current_row)

    if len(item_position_list) > 0:
        print(f'list ITEM found: {item_position_list}')

    board_matrix_np = np.array(board_matrix)
    
    return board_matrix_np

def click_cell(col, row, button='left'):
    """Simulate clicking on a specific cell."""
    x = BOARD_TOP_LEFT[0] + col * CELL_SIZE + CELL_SIZE // 2
    y = BOARD_TOP_LEFT[1] + row * CELL_SIZE + CELL_SIZE // 2
    pyautogui.moveTo(x, y, duration=round(random.uniform(0, 0.1), 2))
    pyautogui.click(x, y, button=button)
##############################################

class MinesweeperHelper:
    ''' Class wth a few helper method to handle n-dimensional
    minesweeper boards: list of neighbors for each cell,
    iteration over all cells etc
    '''

    def __init__(self, shape, wrap_around=False):
        ''' Shape: list of sizes for dimensions (length of the list
        will define how many dimensions the game has)
        Wrap around: if opposite ends of the field are wrapped around
        (Surface of a torus for 2D game)
        '''
        self.shape = shape

        self.wrap_around = wrap_around

        # This is just a dict to store all the neighboring coordinates
        # for all cells, so we won't have to recalculate them every move
        self.neighbors_cache = {}

        # Cache for the list of all iterations
        self.all_iterations_cache = None

    def iterate_over_all_cells(self):
        ''' Returns a list
        [(0, .., 0), (0, .., 1) .. (d1-1, .., dn-1)]
        of all possible coordinates.
        Kind of like "range" but for D-dimensional array of cells.
        '''
        # Serve from cache, if available
        if self.all_iterations_cache is not None:
            return self.all_iterations_cache

        permutations = []

        # We'll have as many permutations as product of all dimensions
        for permutation_number in range(np.prod(self.shape)):

            # List to construct a permutation in
            this_permutation = []
            # And a copy of permutation's cardinal number
            # (need a copy, as we will mutate  it)
            remaining_number = permutation_number

            # Go from back to front through the dimension
            for pos in reversed(self.shape):

                # It is somewhat similar to base conversion,
                # except base changes for each place.

                # This permutation is just a remainder of division
                this_permutation.append(remaining_number % pos)
                # But you need to make sure you subtract by
                # the number in the latest position
                remaining_number -= this_permutation[-1]
                # and divide by the latest base
                remaining_number //= pos

            # Reverse the resulting list (as we started from the right side,
            # the smallest digits), and store it in the final list
            this_permutation.reverse()
            permutations.append(tuple(this_permutation))

        self.all_iterations_cache = permutations
        return permutations

    def valid_coords(self, cell):
        ''' Check if cell's coordinates are valid
        (do't go over field's size).
        Return Tru / False
        '''
        for i, dimension_size in enumerate(self.shape):
            if cell[i] < 0 or cell[i] >= dimension_size:
                return False
        return True

    def cell_surroundings(self, cell):
        ''' Returns a list of coordinates of neighbors of a cell
        taking borders into account
        '''
        # Dynamic programming: use buffer if the result is there
        if cell in self.neighbors_cache:
            return self.neighbors_cache[cell]

        surroundings = []

        # This is to calculate offset. But done outside of for
        # as this is the same for all iterations
        powers = {j: 3**j for j in range(len(self.shape))}

        # Iterate over 3 ** 'N of dimensions' of potential neighbors
        for i in range(3 ** len(self.shape)):

            # Way to calculate all (1, -1, 0) for this permutation
            offset = tuple((i // powers[j]) % 3 - 1
                           for j in range(len(self.shape)))

            # If it is the cell itself - skip
            if offset.count(1) == 0 and offset.count(-1) == 0:
                continue

            # Different ways of calculating neighbors' coordinates
            # with and without wrapping around
            if self.wrap_around:
                cell_with_offset = tuple((cell[i] + offset[i] +
                                          self.shape[i]) % self.shape[i]
                                         for i in range(len(self.shape)))
            else:
                cell_with_offset = tuple(cell[i] + offset[i]
                                         for i in range(len(self.shape)))

            # If resulting coords are valid: add them to the list of neighbors
            if self.valid_coords(cell_with_offset):
                surroundings.append(cell_with_offset)

        # Store in buffer, for future reuse
        self.neighbors_cache[cell] = surroundings

        return surroundings

    def random_coords(self):
        ''' Generates a tuple of self.size random coordinates
        each within the appropriate dimension's size. Returns a tuple
        '''
        coordinates = []
        for dimension_size in self.shape:
            coordinates.append(random.randint(0, dimension_size - 1))
        return tuple(coordinates)

    def are_all_covered(self, field):
        ''' Are all cells covered?
        (to indicate the this is the very first move)
        '''
        for cell in self.iterate_over_all_cells():
            if field[cell] != CELL_COVERED:
                return False
        return True


def bring_to_front(a, b, c):
    # For list a
    for element in c:
        if element in a:
            a.remove(element)
            a.insert(0, element)
    
    # For list b
    for element in c:
        if element in b:
            b.remove(element)
            b.insert(0, element)
    
    return a, b

class MinesweeperGame:
    ''' Class for a minesweeper game: generate game board,
    accept a move, revealed the result, etc
    '''

    def __init__(self, settings=GAME_DOTA_LVL_1, seed=None, field_str=None):
        ''' Initiate a new game: generate mines, calculate numbers.
        Inputs:
        - settings: GameSettings objects with dimensions of the game
                    an the number of mines
        - seed: Seed to use for generating board. None for random.
        - field_str: pre-generated board to use. String with "*" for mines
        '''

        # Shape, a tuple of dimension sizes
        self.shape = settings.shape

        # Make sure there no more mines than cells minus one
        self.mines = min(settings.mines, np.prod(self.shape) - 1)

        # Wrap around option
        self.wrap_around = settings.wrap_around

        # Counter of remaining mines (according to what plays
        # marked as mines, can be inaccurate).
        self.remaining_mines = self.mines

        # Use seed, if seed passed
        if seed is not None:
            random.seed(seed)

        self.helper = MinesweeperHelper(self.shape,
                                        wrap_around=self.wrap_around)


        #Update board info
        self.update_board_settings()
        printBoardInfo()

        # Capture screen then convert to np array
        while True:
            board_image = screenshot_board()
            board_matrix_np = convert_board_capture_to_matrix(board_image)
            if np.any(board_matrix_np == '?') == False:
                break
            else:
                print('capture again')

        self.uncovered = board_matrix_np.T
        self.field = board_matrix_np.T
        print('Load map from dota:')
        print(self.uncovered)
        # print(type(self.uncovered))
        # quit()
        # Default status
        self.status = STATUS_ALIVE

    def update_board_settings(self):
        global BOARD_TOP_LEFT, BOARD_SIZE_COL, BOARD_SIZE_ROW
        if self.shape == (9, 9):
            BOARD_TOP_LEFT = (OFFSET_X_LVL_1, OFFSET_Y_LVL_1)
            BOARD_SIZE_COL = self.shape[0]
            BOARD_SIZE_ROW = self.shape[1]
        elif self.shape == (12, 11):
            BOARD_TOP_LEFT = (OFFSET_X_LVL_2, OFFSET_Y_LVL_2)
            BOARD_SIZE_COL = self.shape[0]
            BOARD_SIZE_ROW = self.shape[1]
        elif self.shape == (15, 13):
            BOARD_TOP_LEFT = (OFFSET_X_LVL_3, OFFSET_Y_LVL_3)
            BOARD_SIZE_COL = self.shape[0]
            BOARD_SIZE_ROW = self.shape[1]
        elif self.shape == (18, 14):
            BOARD_TOP_LEFT = (OFFSET_X_LVL_4, OFFSET_Y_LVL_4)
            BOARD_SIZE_COL = self.shape[0]
            BOARD_SIZE_ROW = self.shape[1]
        elif self.shape == (20, 16):
            BOARD_TOP_LEFT = (OFFSET_X_LVL_5, OFFSET_Y_LVL_5)
            BOARD_SIZE_COL = self.shape[0]
            BOARD_SIZE_ROW = self.shape[1]
        else:
            print(f"unkown pos and size board, self.shape: {self.shape}")

    def handle_mine_click(self, cell):
        ''' Mine (right) click. Simply mark cell a mine
        '''
        # Ignore invalid coordinates
        if not self.helper.valid_coords(cell):
            return

        if self.uncovered[cell] == CELL_COVERED:
            click_cell(cell[0], cell[1], 'right')
            self.remaining_mines -= 1

    def handle_safe_click(self, cell):
        ''' Safe (left) click. Explode if a  mine,
        uncover if not. (includes flood fill for 0)
        '''
        # Ignore invalid coordinates
        if not self.helper.valid_coords(cell):
            return

        # This cell has been clicked on before: ignore it
        if self.uncovered[cell] != CELL_COVERED:
            return

        click_cell(cell[0], cell[1], 'left')

    def move_mouse_to_corner(self):
        # Get the current mouse position
        x, y = pyautogui.position()
        # Determine where to move the mouse based on its current sector
        if x < half_width and y < half_height:
            # print("Mouse is in the top-left sector")
            target_x = BOARD_TOP_LEFT[0] - 50
            target_y = BOARD_TOP_LEFT[1] - 50
        elif x >= half_width and y < half_height:
            # print("Mouse is in the top-right sector")
            target_x = BOARD_TOP_LEFT[0] + CELL_SIZE * BOARD_SIZE_COL + 50
            target_y = BOARD_TOP_LEFT[1] - 50
        elif x < half_width and y >= half_height:
            # print("Mouse is in the bottom-left sector")
            target_x = BOARD_TOP_LEFT[0] - 50
            target_y = BOARD_TOP_LEFT[1] + CELL_SIZE * BOARD_SIZE_ROW + 50
        else:
            # print("Mouse is in the bottom-right sector")
            target_x = BOARD_TOP_LEFT[0] + CELL_SIZE * BOARD_SIZE_COL + 50
            target_y = BOARD_TOP_LEFT[1] + CELL_SIZE * BOARD_SIZE_ROW + 50
        # Move the mouse to the calculated target position with a random duration
        pyautogui.moveTo(target_x, target_y, duration=round(random.uniform(0.1, 0.3), 2))

    def make_a_move(self, safe=None, mines=None):
        ''' Do one minesweeper iteration.
        Accepts list of safe clicks and mine clicks.
        Returns self.status: 0 (not dead), 1 (won), 2 (dead)
        '''
        global item_position_list

        bring_to_front(safe, mines, item_position_list)

        if safe:
            for cell in safe:
                self.handle_safe_click(cell)
        if mines:
            # Mark all the mines
            for cell in mines:
                self.handle_mine_click(cell)

        self.move_mouse_to_corner()

        while True:
            board_image = screenshot_board()
            board_matrix_np = convert_board_capture_to_matrix(board_image)
            if np.any(board_matrix_np == '?') == False:
                break
            else:
                print('capture again')

        self.uncovered = board_matrix_np.T
        self.field = board_matrix_np.T

        return self.status

        # ''' Visual representation of the 1D field
        # Done by converting it to a 2D field with height 1
        # '''
        # width = field_to_show.shape[0]
        # field_to_show = np.reshape(field_to_show, (width, 1))

        # return self.field2str_2d(field_to_show, show_ruler)

    @staticmethod
    def field2str_2d(field_to_show, show_ruler=True):
        ''' Visual representation of the 2D field
        '''
        height = field_to_show.shape[1]
        width = field_to_show.shape[0]

        # Add all data to this string
        output = ""

        # Top ruler + shift fot the top border
        if show_ruler:
            output += " " * 5
            # Top ruler shows only second digit for numbers ending 1..9
            # but both digits for those ending in 0 etc
            output += "".join([f"{i}" if i % 10 == 0 and i != 0
                               else f"{i % 10} "
                               for i in range(width)])
            output += " \n"
            output += " " * 3

        # Top border
        output += "-" * (width * 2 + 3) + "\n"

        # Iterate over all cells, row by row
        for row in range(height):

            # Left border
            if show_ruler:
                output += f"{row:2} "
            output += "! "

            # Iterate over each cell in a row
            for col in range(width):

                cell_value = field_to_show[col, row]

                # Display the character according to characters_to_display
                output += LEGEND[cell_value] + " "

            # Right border
            output += "!\n"

        # Bottom border
        if show_ruler:
            output += " " * 3
        output += "-" * (width * 2 + 3) + "\n"

        return output


        ''' Visual representation of the 4D field
        '''
        fourth = field_to_show.shape[0]
        output = ""
        for i in range(fourth):

            # Show ruler only for the first row
            show_ruler = i == 0

            # Get the 3D representation for the current row
            fields = self.field2str_3d(field_to_show[i],
                                       show_ruler=show_ruler)

            # Go through the 3D line by line
            cur_line_of_fields = fields.split("\n")[:-1]
            for line_n, line in enumerate(cur_line_of_fields):

                # Inserting either a row number or space
                if line_n == len(cur_line_of_fields) // 2 + \
                   (1 if i == 0 else 0):
                    output += str(i)
                else:
                    output += " "
                output += line + "\n"

        return output

    def field2str(self, field_to_show, show_ruler=True):
        return self.field2str_2d(field_to_show, show_ruler)

    def __str__(self):
        ''' Display uncovered part of the board
        '''
        return self.field2str(self.uncovered)

    def parse_input(self, string):
        ''' For Command line play: parse string as
        [x y] | [M x y] | [A x y] ...
        - where x and y are 0-based coordinates
        - "M" to mark the mine
        - "A" is open all around (like 2-button click)
        returns two lists that can be fed right into "make_a_move"
        '''
        safe, mines = [], []
        mode, cell = None, []

        # Break the string by spaces
        for chunk in string.upper().split(" "):

            # We have a coordinate
            if chunk.isnumeric():
                # Add it to the coordinates list
                cell.append(int(chunk))

                # if coords is long enough
                if len(cell) == len(self.shape):

                    # It supposed to be a tuple
                    cell = tuple(cell)

                    # Append the lists accordingly
                    # Single mine
                    if mode == "M":
                        mines.append(cell)
                    # Open around, add all surroundings
                    elif mode == "A":
                        safe.extend(self.helper.cell_surroundings(cell))
                    # Single safe
                    else:
                        safe.append(cell)
                    mode, cell = False, []

            # M and A modify the behavior
            elif chunk in ("M", "A"):
                mode = chunk

        return safe, mines

def main():
    ''' Some tests for minesweeper sim
    '''

    # Seed to generate the game (None for random)
    seed = None

    game = MinesweeperGame(settings=GAME_TEST, seed=seed)

    # For debugging: check out the field
    # print(game.field2str(game.field))

    # Keep making moves, while alive
    while game.status == STATUS_ALIVE:

        # Input the move data
        string = input("Move: ")
        safe, mines = game.parse_input(string)
        game.make_a_move(safe, mines)

        # Display the results
        print(game)
        print(f"Status: {STATUS_MESSAGES[game.status]}")
        print(f"Remaining mines: {game.remaining_mines}")

    print("Thank you for playing!")


if __name__ == "__main__":
    main()

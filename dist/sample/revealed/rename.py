import os

def rename_png_files():
    # Get a list of all .png files in the current directory
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    
    # Sort the list to ensure the files are renamed in a consistent order
    png_files.sort()

    for index, filename in enumerate(png_files, start=1):
        new_index = index
        new_name = f'revealed_{new_index}.png'
        
        # Increment the index until a non-existing file name is found
        while os.path.exists(new_name):
            new_index += 1
            new_name = f'revealed_{new_index}.png'
        
        os.rename(filename, new_name)
        print(f'Renamed {filename} to {new_name}')

if __name__ == "__main__":
    rename_png_files()

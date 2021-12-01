import os

if __name__ == '__main__':
    path = os.path.abspath("./data")
    for dir_path, dir_name file_name in os.walk(path):
        print(file_name)
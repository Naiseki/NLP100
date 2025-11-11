import sys
import numpy as np

def split_file(file, n):
    lines = file.readlines()
    return np.array_split(lines, n)

def main():
    filename = sys.argv[1]
    n = 10
    parts = []
    with open(filename, "r") as file:
        parts = split_file(file, n)
    for i in range(len(parts)):
        with open(f"{filename}_{i}.txt", "w") as out_file:
            out_file.writelines(parts[i])
            


if __name__ == "__main__":
    main()

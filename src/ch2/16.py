import sys
import random

def randomize_file(file):
    lines = file.readlines()
    random.shuffle(lines)
    return lines

def main():
    filename = sys.argv[1]
    result = []
    with open(filename, "r") as file:
        result = randomize_file(file)
        print("".join(result), end="")
            


if __name__ == "__main__":
    main()

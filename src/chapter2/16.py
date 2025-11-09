import sys
import random

def randomize_file(file):
    lines = file.readlines()
    random.shuffle(lines)
    return lines

    

def main():
    filename = sys.argv[1]
    result = []
    with open(filename, 'r') as file:
        result = randomize_file(file)
    with open(f'output/16_output.txt', 'w') as out_file:
        out_file.writelines(result)
            


if __name__ == '__main__':
    main()

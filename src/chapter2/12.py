import sys
from collections import deque

def print_tail(file, n):
    lines = deque(file, maxlen=n)
    for line in lines:
        print(line, end='')

def main():
    filename = sys.argv[1]
    n = int(sys.argv[2])
    with open(filename, 'r') as file:
        print_tail(file, n)


if __name__ == '__main__':
    main()

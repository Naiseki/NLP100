import sys

def print_head(file, n):
    for i in range(n):
        line = file.readline()
        if line is None:
            break
        print(line, end='')


def main():
    filename = sys.argv[1]
    n = int(sys.argv[2])
    with open(filename, 'r') as file:
        print_head(file, n)


if __name__ == '__main__':
    main()

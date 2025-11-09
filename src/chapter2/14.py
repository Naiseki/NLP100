import sys

def print_head(file, n):
    for i, line in enumerate(file):
        if i >= n:
            break
        cols = line.split('\t')
        print(cols[0])


def main():
    filename = sys.argv[1]
    n = int(sys.argv[2])
    with open(filename, 'r') as file:
        print_head(file, n)


if __name__ == '__main__':
    main()

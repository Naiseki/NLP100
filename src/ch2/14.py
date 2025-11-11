import sys

def print_col1(file, n):
    for i, line in enumerate(file):
        if i >= n:
            break
        cols = line.split("\t")
        print(cols[0])


def main():
    filename = sys.argv[1]
    n = 10
    with open(filename, "r") as file:
        print_col1(file, n)


if __name__ == "__main__":
    main()

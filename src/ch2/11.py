import sys

def print_head(file, n):
    for i, line in enumerate(file):
        if i >= n:
            break
        print(line, end="")


def main():
    filename = sys.argv[1]
    n = 10
    with open(filename, "r") as file:
        print_head(file, n)


if __name__ == "__main__":
    main()

import sys

def count_lines(file):
    return sum(1 for line in file)

def main():
    filename = sys.argv[1]
    with open(filename, "r") as file:
        line_count = count_lines(file)
        print(line_count, filename)


if __name__ == "__main__":
    main()

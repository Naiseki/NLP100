import sys

def replace_tab(file, n):
    for i, line in enumerate(file):
        if i >= n:
            break
        line = line.replace("\t", " ")
        print(line, end="")

def main():
    filename = sys.argv[1]
    n = 10
    with open(filename, "r") as file:
        replace_tab(file, n)


if __name__ == "__main__":
    main()

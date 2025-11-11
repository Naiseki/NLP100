import sys

def get_variety(file):
    col1_set = {l.split("\t")[0] for l in file}
    return len(col1_set)

def main():
    filename = sys.argv[1]
    with open(filename, "r") as file:
        print(get_variety(file))


if __name__ == "__main__":
    main()

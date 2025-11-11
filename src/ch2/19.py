import sys

def get_rows(file):
    lines = file.readlines()
    return [line.split("\t") for line in lines]


def main():
    filename = sys.argv[1]
    rows = []
    with open(filename, "r") as file:
        rows = get_rows(file)
    rows.sort(key=lambda x: int(x[2]), reverse=True)
    for row in rows:
        print("\t".join(row), end='')


if __name__ == "__main__":
    main()

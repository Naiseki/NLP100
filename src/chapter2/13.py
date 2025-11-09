import sys

def replace_tab(file, n):
    for i in range(n):
        line = file.readline()
        if line is None:
            break
        line = line.replace('\t', ' ')
        print(line, end='')


def main():
    filename = sys.argv[1]
    n = int(sys.argv[2])
    with open(filename, 'r') as file:
        replace_tab(file, n)


if __name__ == '__main__':
    main()

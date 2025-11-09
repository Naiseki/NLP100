import sys
from collections import Counter

def get_frequency(file):
    col1_list = [l.split('\t')[0] for l in file]
    return Counter(col1_list)


def main():
    filename = sys.argv[1]
    frequency = None
    with open(filename, 'r') as file:
        frequency = get_frequency(file)
    for elm, cnt in frequency.most_common():
        print(elm)


if __name__ == '__main__':
    main()

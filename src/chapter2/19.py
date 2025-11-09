import sys

def get_cols(file):
    lines = file.readlines()
    return [line.split('\t') for line in lines]


def main():
    filename = sys.argv[1]
    cols = []
    with open(filename, 'r') as file:
        cols = get_cols(file)
    cols.sort(key=lambda x: x[2], reverse=True)
    with open('output_19.txt', 'w') as out_file:
        for col in cols:
            out_file.write('\t'.join(col))


if __name__ == '__main__':
    main()

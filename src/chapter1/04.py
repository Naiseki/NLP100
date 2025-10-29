def main():
    one_char_nums = [1, 5, 6, 7, 8, 9, 15, 16, 19]
    text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
    text_list = text.split()
    result_dict = {}
    for i in range(len(text_list)):
        if i + 1 in one_char_nums:
            result_dict[text_list[i][0]] = i + 1
        else:
            result_dict[text_list[i][0:2]] = i + 1
    print(result_dict)

if __name__ == "__main__":
    main()

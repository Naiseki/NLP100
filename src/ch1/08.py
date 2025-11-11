def cipher(text):
    text_list = list(text)
    for i in range(len(text_list)):
        if text_list[i].islower():
            code = ord(text_list[i])
            text_list[i] = chr(219 - code)
    return ''.join(text_list)

def main():
    print(cipher("Hello World!"))

if __name__ == "__main__":
    main()

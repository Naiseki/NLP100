def main():
    text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    text = text.replace(",", "").replace(".", "")
    text_list = text.split()
    word_lengths = [len(word) for word in text_list]
    print(word_lengths)

if __name__ == "__main__":
    main()

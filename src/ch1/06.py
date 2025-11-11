def ngram(target, n):
    return [target[i:i+n] for i in range(len(target) - n + 1)]

def main():
    x_text = "paraparaparadise"
    y_text = "paragraph"
    target_word = "se"
    
    x = set(ngram(x_text, 2))
    y = set(ngram(y_text, 2))

    print("Xのbi-gram:", x)
    print("Yのbi-gram:", y)
    print("和集合:", x | y)
    print("積集合:", x & y)
    print("差集合:", x - y)
    print(f"Xに'{target_word}'は含まれるか:", target_word in x)
    print(f"Yに'{target_word}'は含まれるか:", target_word in y)

if __name__ == "__main__":
    main()

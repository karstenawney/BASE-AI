text = input("Enter text: ")
alphabet = "abcdefghijklmnopqrstuvwxyz "
translated = []
for character in text:
    if character in alphabet:
        for i, letter in enumerate(alphabet):
            if letter == character:
                translated.append(i)
    else:
        translated.append(26)
print (translated)
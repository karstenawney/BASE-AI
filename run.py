from ai import run, load

def main():
    file = input("Enter model path: ")
    model = load(file)
    input_len = int(input("Enter number of inputs: "))
    while True:
        inputs = []
        for i in range(input_len):
            inputs.append(float(input(f"Enter input {i + 1}: ")))
        outputs = run(inputs, model)
        for i in range(len(outputs)):
            print (f"Output {i + 1}: {float(outputs[i])}")

if __name__ == "__main__":
    main()
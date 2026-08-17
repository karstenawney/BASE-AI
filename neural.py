# Note: The numbers in model and modelbias are int16_t, as are activelayer. s is int32_t, clamped to int16_t

model = [[[1]]]
modelbias = [[1]]
activelayer = [1]
for layer, biaslayer in zip(model, modelbias):
    newlayer = [0] * len(layer)
    for i, (neuron, b) in enumerate(zip(layer, biaslayer)):
        s = 0
        for x, w in zip(activelayer, neuron):
            s += (x * w // 16384)
        newlayer[i] = (min(max(0, s + b), 32768))
    activelayer = newlayer
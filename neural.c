#include <stdint.h>

void layer(
    const int16_t * restrict weights,
    const int16_t * restrict bias,
    const int16_t * restrict input,
    int16_t * restrict output,
    int input_length,
    int output_length
) {
    for (int n = 0; n < output_length; n++) {
        int32_t s = 0;
        int weight_offset = input_length * n;

        for (int a = 0; a < input_length; a++) {
            s += ((int32_t)weights[weight_offset + a] * (int32_t)input[a]) >> 14;
        }

        if (s < 0) {
            s = 0;
        } else if (s > 32767) {
            s = 32767;
        }

        output[n] = (int16_t)s;
    }
}
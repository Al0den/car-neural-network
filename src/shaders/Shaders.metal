// https://developer.apple.com/documentation/metal/libraries/building_a_library_with_metal_s_command-line_tools
#include <metal_stdlib>
using namespace metal;

kernel void points_offsets(const device short *input [[ buffer(0) ]], const device uint8_t *track [[ buffer(1) ]], device short *out [[ buffer(2) ]], uint id [[ thread_position_in_grid ]]) {
    short car_x = input[id * 5];
    short car_y = input[id * 5 + 1];
    short direction = input[id * 5 + 2];
    short track_id = input[id * 5 + 3];
    float ppm = (float)(input[id * 5 + 4]) / 1000; // pixels per meter (1000 = 1 meter

    short x = car_x;
    short y = car_y;
    short distance = 0;

    if (track_id == -1) {
        out[id] = 0;
        return;
    }

    float pi = 3.1415;
    float cosinus = (float)cos((float)direction * pi/180);
    float sinus = (float)sin((float)direction * pi/180);
    
    int max_distance = (int)(200 * ppm);
              
    int point_search_jump = 25;

    if (track[track_id * 5000 * 5000 + y * 5000 + x] == 0) {
        out[id] = distance;
        return;
    }
    
    for (int i = 0; i < max_distance; i += point_search_jump) {
        x = car_x + (int)(i * sinus);
        y = car_y + (int)(i * cosinus);
        if (track[track_id * 5000 * 5000 + y * 5000 + x] == 0) {
            distance = i;
            break;
        }
        distance = i;
    }
    if (distance == max_distance) {
        out[id] = distance;
        return;
    }
                                
    int jump = point_search_jump / 2;
    int rep = 0;
    while (jump > 0 && rep < 30) {
        if (track[track_id * 5000 * 5000 + y * 5000 + x] == 0) {
            distance -= jump;
        } else {
            distance += jump;
        }
        jump *= 0.5;
        rep++;
        x = car_x + (int)(distance * sinus);
        y = car_y + (int)(distance * cosinus);
    }
    out[id] = distance;
    return;
}

kernel void dot_product(const device int *input [[ buffer(0) ]], const device float *weights [[ buffer(1) ]], device float *out [[ buffer(2) ]], uint id [[ thread_position_in_grid ]]) {
    int input_size = input[1];
    int a_start_index = input[id * 2 + 2];
    int b_start_index = input[id * 2 + 3];

    float sum = 0;
    for (int i = 0; i < input_size; i++) {
        sum += weights[a_start_index + i] * weights[b_start_index + i];
    }
    out[id] = sum;
    return;
}
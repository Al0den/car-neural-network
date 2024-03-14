// https://developer.apple.com/documentation/metal/libraries/building_a_library_with_metal_s_command-line_tools
#include <metal_stdlib>
using namespace metal;


kernel void points_offsets(const device short *input [[ buffer(0) ]], const device uint8_t *track [[ buffer(1) ]], device short *out [[ buffer(2) ]], const device short *offsets [[ buffer(3) ]], uint id [[ thread_position_in_grid ]]) {
    int num_offsets = 15;
    
    int car_id = int(id / num_offsets);
    int offset_id = int(id % num_offsets);

    short offset = offsets[offset_id];

    short car_x = input[car_id * 10];
    short car_y = input[car_id * 10 + 1];
    short direction = (input[car_id * 10 + 2] + 90 + offset) % 360;
    short track_id = input[car_id * 10 + 3];
    float ppm = (float)(input[car_id * 10 + 4]) / 1000; // pixels per meter (1000 = 1 meter)
    int track_offset = track_id * 5000 * 5000;

    if (car_x == -1) {
        return;
    }
    out[id] = 0;

    short x = car_x;
    short y = car_y;
    short distance = 0;

    float angleInRadians = (float)direction * (M_PI_F / 180.0f);
    float cosinus;
    float sinus = sincos(angleInRadians, cosinus);

    for(int i=0; i<int(30*ppm); i++) {
        x = car_x + (int)(i * sinus);
        y = car_y + (int)(i * cosinus);
        if (track[track_offset + y * 5000 + x] != 0) {
            distance = i;
            break;
        }
        distance = i;
    }

    if (track[track_offset + y * 5000 + x] == 0) {
        out[id] = 0;
        return;
    }
    
    int max_distance = (int)(200 * ppm);
              
    int point_search_jump = 25;
    
    for (int i = distance; i < max_distance; i += point_search_jump) {
        x = car_x + (int)(i * sinus);
        y = car_y + (int)(i * cosinus);
        if (track[track_offset + y * 5000 + x] == 0) {
            distance = i;
            break;
        }
        distance = i;
    }

    while (track[track_offset + y * 5000 + x] == 0) {
        distance -= 1;
        x = car_x + (int)(distance * sinus);
        y = car_y + (int)(distance * cosinus);
    }

    out[id] = distance;
    return;
}

kernel void update_car(const device short *input [[ buffer(0) ]], const device uint8_t *track [[ buffer(1) ]], device short *out [[ buffer(2) ]], uint id [[ thread_position_in_grid ]]) {
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
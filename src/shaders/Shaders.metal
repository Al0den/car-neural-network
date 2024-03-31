// https://developer.apple.com/documentation/metal/libraries/building_a_library_with_metal_s_command-line_tools
#include <metal_stdlib>
using namespace metal;

kernel void points_offsets(const device short *input [[ buffer(0) ]], const device uint8_t *track [[ buffer(1) ]], device float *out [[ buffer(2) ]], const device short *offsets [[ buffer(3) ]], uint id [[ thread_position_in_grid ]]) {
    int num_offsets = 15;
    int max_points_distance = 200;

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

    out[id] = min(1.0, distance/ (ppm * max_points_distance));
    return;
}

kernel void process_corner(const device short *input [[ buffer(0) ]], const device short *corner_data [[ buffer(4) ]], device float *out [[ buffer(5) ]], uint id [[ thread_position_in_grid ]]) {
    int num_corner_data = 4;
    int num_corners = 2;
    
    int max_corner_distance = 800;
    
    int car_id = id / (num_corners);
    int corner_id = id * num_corner_data;
    
    short corner_x = corner_data[corner_id];
    short corner_y = corner_data[corner_id+ 1];
    short corner_dir = corner_data[corner_id + 2];
    short corner_ampl = corner_data[corner_id + 3];

    short car_x = input[car_id * 10];
    short car_y = input[car_id * 10 + 1];
    short car_direction = input[car_id * 10 + 2] ;

    short ppm = (float)(input[car_id * 10 + 4]) / 1000; // pixels per meter (1000 = 1 meter)

    float distance_raw = sqrt((float)(corner_x - car_x) * (float)(corner_x - car_x) + (float)(corner_y - car_y) * (float)(corner_y - car_y));
    float distance = distance_raw / (ppm * max_corner_distance);
    distance = min(distance, (float)1.0);
    float angle = corner_dir - car_direction;
    while(angle > 180) {
        angle -= 360;
    }
    while(angle < -180) {
        angle += 360;
    }
    int left_or_right = angle >= 0 ? 1 : -1;
    float result_ampl = left_or_right * min(1.0, (float)corner_ampl/100);
    out[id * num_corners] = distance;
    out[id * num_corners + 1] = result_ampl;
    return;
}

float dot_product(const device float *arr1, const device float *arr2, int n) {
    float result = 0;
    for (int i = 0; i < n; i++) {
        result += arr1[i] * arr2[i];
    }
    return result;
}

float tanh(float x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

float relu(float x) {
    return max(0.0, x);
}
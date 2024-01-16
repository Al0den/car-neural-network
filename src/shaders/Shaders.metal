// https://developer.apple.com/documentation/metal/libraries/building_a_library_with_metal_s_command-line_tools
#include <metal_stdlib>
using namespace metal;

kernel void points_offsets(const device int *input [[ buffer(0) ]], const device int *track [[ buffer(1) ]], device int *out [[ buffer(2) ]], uint id [[ thread_position_in_grid ]]) {
    int car_x = input[id * 4];
    int car_y = input[id * 4 + 1];
    int direction = input[id * 4 + 2];
    int track_id = input[id * 4 + 3];
    if (track_id == -1) {
        out[id * 2] = -1;
        out[id * 2 + 1] = -1;
        return;
    }

    float pi = 3.1415;
    float cosinus = (float)cos((float)direction * pi/180);
    float sinus = (float)sin((float)direction * pi/180);
    
    int distance = 0;
    int max_distance = 200;
    int x = car_x;
    int y = car_y;
                                
    int point_search_jump = 25;

    if (track[track_id * 5000 * 5000 + y * 5000 + x] == 0) {
        out[id * 2] = x;
        out[id * 2 + 1] = y;
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
        out[id * 2] = x;
        out[id * 2 + 1] = y;
        return;
    }
                                
    int jump = point_search_jump / 2;
    int rep = 0;
    while (jump > 0 && rep < 20) {
        if (track[track_id * 5000 * 5000 + y * 5000 + x] == 0) {
            distance -= jump;
        } else {
            distance += jump;
        }
        jump /= 2;
        rep++;
        x = car_x + (int)(distance * sinus);
        y = car_y + (int)(distance * cosinus);
    }
                                
    out[id * 2] = x;
    out[id * 2 + 1] = y;
    return;
}
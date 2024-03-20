import numpy as np
import json
import pygame

from utils import calculate_distance, angle_distance

def smoothen(track):
    print("Smoothening track " + track)
    data = np.load(f"./data/tracks/{track}.npy", allow_pickle=True).item()
    track_matrix = data['track']
    center_mat = np.zeros((track_matrix.shape[1], track_matrix.shape[0], 3))

    center_line = np.argwhere(track_matrix == 10)
    for i, j in center_line:
        center_mat[i][j] = 255

    with open('./src/config.json', 'r') as json_file:
        config_data = json.load(json_file)

    real_width = config_data['real_track_width'].get(track)
    ppm = config_data['pixel_per_meter'].get(track)


    offsets = [(i, j) for i in range(-80, 80) for j in range(-80, 80) if calculate_distance((0, 0), (i, j)) <= (real_width/2 * ppm) and (i, j) != (0, 0)]

    def create_new_matrix(initial_matrix):
        height, width, _ = initial_matrix.shape
        new_matrix = np.zeros((height, width), dtype=np.uint8)  # Create a new matrix initialized with zeros

        indices_255 = np.argwhere(np.all(initial_matrix == 255, axis=-1))  # Check if all channels are 255
        count = 0
        for i, j in indices_255:
            count += 1
            for offset in offsets:
                if 0 <= i + offset[0] < height and 0 <= j + offset[1] < width:
                    new_matrix[i + offset[0]][j + offset[1]] = 255

        return new_matrix

    new_mat = create_new_matrix(center_mat)
    final_mat = np.zeros((track_matrix.shape[1], track_matrix.shape[0]), dtype=np.uint8)
    track_surface = pygame.Surface((track_matrix.shape[1], track_matrix.shape[0]), pygame.SRCALPHA)
    track_surface.fill((0, 100, 0))
    track_surface.set_alpha(None)
    red = (255, 0, 0)
    gray = (100, 100, 100)
    orange = (168, 155, 50)

    non_zero_coords = np.argwhere(new_mat == 255)
    for coord in non_zero_coords:
        i, j = coord[0], coord[1]
        final_mat[i, j] = 1
        if track_matrix[i, j] != 1 and track_matrix[i, j] != 0:
            final_mat[i, j] = track_matrix[i, j]

    non_zero_final_coords = np.argwhere(final_mat != 0)
    for coord in non_zero_final_coords:
        i, j = coord[0], coord[1]
        value = final_mat[i][j]
        if value == 1:
            neighbors = []
            for offset in [(0, 1), (0, -1), (1, 0), (-1, 0), (1,1 ), (1, -1), (-1, 1), (-1, -1)]:
                if 0 <= i + offset[0] < track_matrix.shape[0] and 0 <= j + offset[1] < track_matrix.shape[1]:
                    if final_mat[i + offset[0], j + offset[1]] != 10:
                        neighbors.append(final_mat[i + offset[0], j + offset[1]])
            track_surface.set_at((j, i), gray)
        elif value == 2:
            track_surface.set_at((j, i), orange)
        elif value == 3:
            track_surface.set_at((j, i), red)
        elif value == 10:
            # Get the most present neighbor value in matrix
            neighbors = []
            for offset in [(0, 1), (0, -1), (1, 0), (-1, 0), (1,1 ), (1, -1), (-1, 1), (-1, -1)]:
                if 0 <= i + offset[0] < track_matrix.shape[0] and 0 <= j + offset[1] < track_matrix.shape[1]:
                    if final_mat[i + offset[0], j + offset[1]] != 10:
                        neighbors.append(final_mat[i + offset[0], j + offset[1]])
            mode = max(set(neighbors), key=neighbors.count)
            if mode == 2:
                track_surface.set_at((j, i), orange)
            elif mode == 3:
                track_surface.set_at((j, i), red)
            else:
                track_surface.set_at((j, i), gray)
        elif track_matrix[i][j] > 99:
            track_surface.set_at((j, i), gray)
    corners_raw = data['corners']
    corners = [corner_raw[0] for corner_raw in corners_raw]
    
    for corner in corners:
        # Draw circle
        pygame.draw.circle(track_surface, (255, 0, 0), (corner[0], corner[1]), 3)

    center_line = np.where(track_matrix == 10)
    index = int(np.random.uniform(0, len(center_line[0])))
    start_y, start_x = center_line[0][index], center_line[1][index]
    
    current_length = 0
    drawing_line = True

    x, y = start_x, start_y
    direction = 0
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for offset in offsets:
        if track_matrix[y + offset[0], x + offset[1]] == 10:
            direction = np.degrees(np.arctan2(-offset[0], offset[1]))
            break
    
    started = False
    seen = 0
    ppm = config_data['pixel_per_meter'].get(track)
    while (start_y, start_x) != (y, x) or not started:
        started = True
        print(f"{seen} / {len(center_line[0])}")
        potential_offsets = [offset for offset in offsets if track_matrix[y + offset[0], x + offset[1]] == 10 and angle_distance(np.degrees(np.arctan2(-offset[0], offset[1])), direction) <= 110]
        if len(potential_offsets) == 0:
            two_wide = [(0,2), (0,-2), (2,0), (-2,0), (2,2), (2,-2), (-2,2), (-2,-2), (2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
            potential_offsets = [offset for offset in two_wide if track_matrix[y + offset[0], x + offset[1]] == 10 and angle_distance(np.degrees(np.arctan2(-offset[0], offset[1])), direction) <= 110]
        if len(potential_offsets) == 0:
            print("No potential offsets")
            break
        
        offset = potential_offsets[np.random.randint(0, len(potential_offsets))]
        x, y = x + offset[1], y + offset[0]
        seen += 1
        direction = np.degrees(np.arctan2(-offset[0], offset[1]))
        if drawing_line:
            track_surface.set_at((x, y), (255, 255, 255))
        current_length += 1
        if current_length >= 3 * ppm and drawing_line:
            drawing_line = False
            current_length = 0
        elif current_length >= 10 * ppm and not drawing_line:
            drawing_line = True
            current_length = 0

    all_end_positions = np.argwhere(track_matrix == 3)
    end_pos_x, end_pos_y = 0, 0
    for end_pos in all_end_positions:
        end_pos_x += end_pos[1]
        end_pos_y += end_pos[0]
    end_pos_x /= len(all_end_positions)
    end_pos_y /= len(all_end_positions)
    print(end_pos_x, end_pos_y, track_matrix[int(end_pos_y), int(end_pos_x)])
    assert(track_matrix[int(end_pos_y), int(end_pos_x)] in [3, 10])

    config_data['end_pos'][track] = (end_pos_x, end_pos_y)
    with open('./src/config.json', 'w') as json_file:
        json.dump(config_data, json_file)
    
    data['track'] = final_mat
    pygame.image.save(track_surface, "./data/tracks/" + track + "_surface.png")
    np.save(f"./data/tracks/{track}.npy", data)

if __name__=="__main__":
    track = input("Track to smoothen: ")
    smoothen(track)
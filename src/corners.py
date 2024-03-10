import numpy as np
import json

def get_corners(track, track_name, threshold=65, data_size=50):
    points = np.where(track == 10)
    with open('./src/config.json', 'r') as json_file:
        config_data = json.load(json_file) 

    offsets = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]

    def angle_distance(angle1, angle2):
        val = abs(angle1 - angle2) % 360
        if abs(angle1 - angle2) > 180:
            return 360 - val
        return val

    def distance_to_line(point, line_start, line_end):
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        return np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2 + 0.01)

    def get_point_further(x, y, direction):
        current_x, current_y = x, y
        current_dir = direction
        ppm = config_data['pixel_per_meter'].get(track_name)
        max_dist = int(data_size * ppm)
        for i in range(max_dist):
            valid_offsets = [offset for offset in offsets if track[current_y + offset[1], current_x + offset[0]] == 10 and angle_distance(current_dir, np.degrees(np.arctan2(-offset[1], offset[0]))) <= 110]
            if len(valid_offsets) == 0:
                two_wide = [(0,2), (0,-2), (2,0), (-2,0), (2,2), (2,-2), (-2,2), (-2,-2), (2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
                valid_offsets = [offset for offset in two_wide if track[current_y + offset[0], current_x + offset[1]] == 10 and angle_distance(np.degrees(np.arctan2(-offset[0], offset[1])), current_dir) <= 110]
            if len(valid_offsets) == 0:
                print("No potential offsets")
                break
            chosen_offset = valid_offsets[0]
            chosen_angle = np.degrees(np.arctan2(-chosen_offset[1], chosen_offset[0]))
            
            # Update current position and direction
            current_dir = chosen_angle
            current_x += chosen_offset[0]
            current_y += chosen_offset[1]

        return current_x, current_y

    per_point_data = np.zeros((5000, 5000))

    for i in range(len(points[0])):
        print(f"Currently on point: {i}/{len(points[0])}    \r", end='', flush=True)
        x, y = points[1][i], points[0][i]
        valid_offsets = [offset for offset in offsets if track[y + offset[1], x + offset[0]] == 10]
        if not valid_offsets: continue
        valid_directions = [np.degrees(np.arctan2(-offset[1], offset[0])) for offset in valid_offsets]
        valid_directions = valid_directions[:2]
        
        if len(valid_directions) != 2: continue
        forward = get_point_further(x, y, valid_directions[0])
        backward = get_point_further(x, y, valid_directions[1])
        per_point_data[y, x] = distance_to_line((x, y), forward, backward)

    # For every pixel in track_pixels, set its value to the value of its nearest point in per_point_data within a maximum distance of 80
    max_distance = 10
    i = 0
    final_data = np.zeros((5000, 5000))

    for i in range(len(points[0])):
        print(f"Currently on point: {i}/{len(points[0])}    \r", end='', flush=True)
        for x in range(points[1][i] - max_distance, points[1][i] + max_distance):
            for y in range(points[0][i] - max_distance, points[0][i] + max_distance):
                if np.sqrt((x - points[1][i])**2 + (y - points[0][i])**2) <= max_distance:
                    final_data[y, x] = per_point_data[points[0][i], points[1][i]]

    corners = np.where(per_point_data >= threshold)

    blob_distance = 50
    blobs = []

    # For every corner, check if it is within blob_distance of any other corner. If it is, add it to the blob
    for i in range(len(corners[0])):
        print(f"Currently on corner: {i}/{len(corners[0])}    \r", end='', flush=True)
        x, y = corners[1][i], corners[0][i]
        found_blob = False
        for blob in blobs:
            for blob_point in blob:
                if np.sqrt((x - blob_point[0])**2 + (y - blob_point[1])**2) <= blob_distance:
                    blob.append((x, y))
                    found_blob = True
                    break
            if found_blob: break
        if not found_blob: blobs.append([(x, y)])

    # Check randomly to see if any blobs are within blob_distance of each other. If they are, merge them

    p = 0
    while True:
        p += 1
        print(f"Currently on pass: {p}             \r", end='', flush=True)
        changed = False
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                for blob_point in blobs[i]:
                    for other_blob_point in blobs[j]:
                        if np.sqrt((blob_point[0] - other_blob_point[0])**2 + (blob_point[1] - other_blob_point[1])**2) <= blob_distance:
                            blobs[i] = blobs[i] + blobs[j]
                            blobs.pop(j)
                            changed = True
                            break
                    if changed: break
                if changed: break
            if changed: break
        if not changed: break

    kept_corners = []
    # Keep the value per blob that minmise the sum of the distance * value to the other points
    for blob in blobs:
        min_value = 100000000
        min_point = None
        for point in blob:
            value = 0
            for other_point in blob:
                value += np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2) * per_point_data[other_point[1], other_point[0]]
            if value < min_value:
                min_value = value
                min_point = point
        kept_corners.append(min_point)

    # For every corner, iterate through the next 10 track == 10 to figure out if its a turning right or turning left corner
    
    return kept_corners, corners, final_data

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    track_name = input("Track name: ")
    track = np.load(f"./data/tracks/{track_name}.npy", allow_pickle=True).item()['track']
    corners, all_corners, final_data = get_corners(track, track_name)
    x = [i[0] for i in corners]
    y = [i[1] for i in corners]
    all_x = all_corners[1]
    all_y = all_corners[0]
    plt.imshow(final_data, cmap='gray')
    plt.scatter(all_x, all_y, c='b', s=3)
    plt.scatter(x, y, c='r', s=5)
    
    plt.show()

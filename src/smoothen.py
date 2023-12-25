import numpy as np
import pygame

from precomputed import real_track_width, pixel_per_meter
from utils import calculate_distance

def main(track):
    data = np.load(f"./data/tracks/{track}.npy", allow_pickle=True).item()
    track_matrix = data['track']
    center_mat = np.zeros((track_matrix.shape[1], track_matrix.shape[0], 3))

    center_line = np.argwhere(track_matrix == 10)
    for i, j in center_line:
        center_mat[i][j] = 255

    offsets = [(i, j) for i in range(-80, 80) for j in range(-80, 80) if calculate_distance((0, 0), (i, j)) <= (real_track_width[track]/2 * pixel_per_meter[track]) and (i, j) != (0, 0)]

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
            if 0 in neighbors:
                track_surface.set_at((j, i), (0, 0, 0))
            else:
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

    data['track'] = final_mat
    pygame.image.save(track_surface, "./data/tracks/" + track + "_surface.png")
    np.save(f"./data/tracks/{track}.npy", data)

if __name__=="__main__":
    track = input("Track to smoothen: ")
    main(track)
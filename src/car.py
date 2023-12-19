import pygame
import numpy as np
import random

from utils import calculate_distance, next_speed, angle_distance, new_brake_speed
from precomputed import sin, cos, potential_offsets_for_angle, points_offset, pixel_per_meter, offsets, directions
from settings import *

class Car:
    def __init__(self, track, start_pos, start_dir, track_name=None):
        self.track = track
        self.track_name = track_name or False

        self.ppm = pixel_per_meter[self.track_name]
        
        self.x = start_pos[1]
        self.y = start_pos[0]
        self.direction = start_dir

        self.start_x = self.x
        self.start_y = self.y
        self.start_direction  = self.direction
        self.previous_center_line = (self.x, self.y)
        self.score = 0

        self.front_left, self.front_right, self.back_left, self.back_right = [0, 0], [0, 0], [0, 0], [0, 0]
        self.acceleration, self.brake, self.speed, self.steer = 0, 0, 0, 0
        self.lap_times, self.lap_time, self.laps = [], 0, 0
        self.checkpoints_seen, self.checkpoints = [], []
        self.previous_points = [None] * len(points_offset)

        self.update_corners()

        self.died = False
        
    def tick(self, game):
        self.applyPlayerInputs()
        self.updateCar()
        return self.checkCollisions(game.ticks)

    def checkCollisions(self, ticks):
        toCheck = [self.front_left, self.front_right, self.back_left, self.back_right]
        count = 0
        # Check that atleast 1 wheel is on track
        for point in toCheck:
            if self.track[int(point[1]), int(point[0])] == 0:
                count += 1
            else:
                break
        if count > 3 and not god:
            self.kill()
            return False
        # Checkpoints and end of lap
        if self.track[int(self.y), int(self.x)] == 3:
            self.lap_time = ticks
            self.kill()
            return False
        elif self.track[int(self.y), int(self.x)] == 2:
            seen = False
            for checkpoint in self.checkpoints_seen:
                if calculate_distance(checkpoint, (self.x, self.y)) < min_checkpoint_distance:
                    seen = True
                    seen_time = checkpoint[2]
            if seen == False:
                self.checkpoints_seen.append((self.x, self.y, ticks))
            elif ticks - seen_time > max_time_on_checkpoint:
                self.kill()
                return False
        return True
    
    def kill(self):
        self.died = True
        self.x = self.start_x
        self.y = self.start_y
        self.direction = self.start_direction
        self.speed = 0
        self.acceleration = 0
        self.brake = 0
        self.steer = 0
        self.start_ticks = 0
        self.checkpoints_seen = []

    def accelerate(self):
        self.acceleration += acceleration_increment
        self.acceleration = min(1, self.acceleration)

    def slow_down(self):
        self.brake += brake_increment
        self.brake = min(1, self.brake)

    def update_steer(self, value):
        if value > 0:
            self.steer += steer_increment
            self.steer = min(1, self.steer)
        elif value < 0:
            self.steer -= steer_increment
            self.steer = max(-1, self.steer)

    def applyPlayerInputs(self):
        keys = pygame.key.get_pressed()
        power, steer, brake = False, False, False
        if keys[pygame.K_w]:
            self.accelerate()
            power = True
        if keys[pygame.K_s]:
            self.slow_down()
            brake = True
        if keys[pygame.K_a]:
            self.update_steer(1)
            steer = True
        if keys[pygame.K_d]:
            self.update_steer(-1)
            steer = True
       
        self.checkForAction(brake, power, steer)
        
    def applyAgentInputs(self, action):
        power = action[0]
        steer = action[1]
        steer_change, brake_change, power_change = False, False, False
        if power > 0.5:
            self.accelerate()
            power_change = True
        if power < -0.5:
            self.slow_down()
            brake_change = True
        if steer > 0.5:
            self.update_steer(1)
            steer_change = True
        if steer < -0.5:
            self.update_steer(-1)
            steer_change = True
        self.checkForAction(brake_change, power_change, steer_change)
    
    def checkForAction(self, brake, power, steer):
        if brake == False:
            self.brake -= delta_t * 3
            self.brake = max(0, self.brake)
        if power == False:
            self.acceleration -= delta_t * 3
            self.acceleration = max(0, self.acceleration)
        if steer == False:
            if(self.steer > 0):
                self.steer -= delta_t * 3
                self.steer = max(0, self.steer)
            elif self.steer < 0:
                self.steer += delta_t * 3
                self.steer = min(0, self.steer)
        if self.steer > 1:
            self.steer = 1
        if self.steer < -1:
            self.steer = -1
        
    def updateCar(self):     
        wheel_angle = self.steer * 14 
        speed_factor = max(1.0 - self.speed / (max_speed + 20), 0.1)

        wheel_angle *= speed_factor
        if wheel_angle != 0:
            turning_radius = car_length * self.ppm / np.tan(np.radians(wheel_angle))
        else:
            turning_radius = float('inf')  # Ligne droite
        inner_steer_angle = np.arctan(car_length * self.ppm / (turning_radius - car_width * self.ppm / 2))
        outer_steer_angle = np.arctan(car_length * self.ppm / (turning_radius - car_width * self.ppm / 2))

        wheel_angle = (inner_steer_angle + outer_steer_angle) / 2.0

        self.direction += wheel_angle * delta_t * (self.speed + 20) * turn_coeff

        # - Car speed
        self.speed += (next_speed(self.speed) - self.speed) * self.acceleration
        if self.brake > 0:
            self.speed += (new_brake_speed(self.speed) - self.speed) * self.brake

        drag_force = 0.5 * drag_coeff * reference_area * self.speed ** 2
        drag_acceleration = drag_force / car_mass
        self.speed -= drag_acceleration * delta_t * (1-self.acceleration) * (1-self.brake)

        displacement = (self.speed / 3.6) * pixel_per_meter[self.track_name]
        self.x += displacement * cos[(int(self.direction) % 360) * 10] * delta_t
        self.y -= displacement * sin[(int(self.direction) % 360) * 10] * delta_t

        self.speed = max(0, min(max_speed, self.speed))
        self.direction %= 360
        self.x = max(0, min(len(self.track[0]) - 1, self.x))
        self.y = max(0, min(len(self.track) - 1, self.y))

        self.update_corners()

    def update_corners(self):
        # Ugly formulas for corners positions, but they work
        half_small_side = car_width * self.ppm / 2
        half_big_side = car_length * self.ppm / 2
        angle = np.arctan(half_small_side / half_big_side)
        cos_1 = cos[(int(self.direction + angle - 90)% 360) * 10]
        sin_1 = sin[(int(self.direction + angle - 90)% 360) * 10]

        self.front_left[0] = self.x - half_small_side * cos_1 - half_big_side * sin_1
        self.front_left[1] = self.y - half_big_side * cos_1 + half_small_side * sin_1
        self.front_right[0] = self.x + half_small_side * cos_1 - half_big_side * sin_1
        self.front_right[1] = self.y - half_big_side * cos_1 - half_small_side * sin_1

        self.back_left[0] = self.x - half_small_side * cos_1 + half_big_side * sin_1
        self.back_left[1] = self.y + half_big_side * cos_1 + half_small_side * sin_1
        self.back_right[0] = self.x + half_small_side * cos_1 + half_big_side * sin_1
        self.back_right[1] = self.y + half_big_side * cos_1 - half_small_side * sin_1

    def getPoint(self, offset):
        dx = 1.0
        angle = int(self.direction + 90 + offset) % 360
        sinus = sin[angle * 10]
        cosinus = cos[angle * 10]
        eval_x, eval_y = dx * sinus + self.x, dx * cosinus + self.y

        distance_from_track = 0
        while int(eval_x) < len(self.track[0]) and int(eval_y) < len(self.track) and self.track[int(eval_y), int(eval_x)] == 0 and distance_from_track < 50:
            distance_from_track += 1
            dx += 1.0
            eval_x, eval_y = dx * sinus + self.x, dx * cosinus + self.y
        
        if int(eval_x) >= len(self.track[0]) or int(eval_y) >= len(self.track):
            return (eval_x, eval_y)
        
        if distance_from_track == 50: return (self.x,self.y)

        i = 0
        while self.track[int(eval_y), int(eval_x)] != 0 and i < max_points_distance * self.ppm:
            dx += point_search_jump
            eval_x, eval_y = dx * sinus + self.x, dx * cosinus + self.y
            i += point_search_jump

        while self.track[int(eval_y), int(eval_x)] == 0:
            dx -= 1.0
            eval_x, eval_y = dx * sinus + self.x, dx * cosinus + self.y
        dx += 1.0
        eval_x, eval_y = dx * sinus + self.x, dx * cosinus + self.y

        return (eval_x, eval_y)
    
    def getPoints(self):
        self.previous_points = [self.getPoint(offset) for offset in points_offset]
        return self.previous_points
    
    def get_centerline(self):
        if self.track[int(self.y), int(self.x)] == 10:
            self.previous_center_line = (int(self.x), int(self.y))
            self.center_line_direction = 0
            return int(self.x), int(self.y)
        for i in range(max_center_line_distance):
            angle = int(self.direction + 90) % 360
            x = self.x + i * cos[int(angle * 10)] * 2
            y = self.y - i * sin[int(angle * 10)] * 2
            if self.validIndex(y, x) and self.track[int(y), int(x)] == 10:
                self.previous_center_line = (int(x), int(y))
                self.center_line_direction = 1
                return int(x), int(y)
            # Check potential neighbors
            for offset in potential_offsets_for_angle[angle]:
                if self.validIndex(x + offset[0], y + offset[1]) and self.track[int(y + offset[1]), int(x + offset[0])] == 10:
                    self.previous_center_line = (int(x + offset[0]), int(y + offset[1]))
                    self.center_line_direction = 1
                    return int(x + offset[0]), int(y + offset[1])
            angle = int(self.direction - 90) % 360
            x = self.x + i * cos[int(angle * 10)] * 2
            y = self.y - i * sin[int(angle * 10)] * 2
            if self.validIndex(x, y) and self.track[int(y), int(x)] == 10:
                self.previous_center_line = (int(x), int(y))
                self.center_line_direction = -1
                return int(x), int(y)
            # Check potential neighbors
            for offset in potential_offsets_for_angle[angle]:
                if self.validIndex(x + offset[0], y + offset[1]) and self.track[int(y + offset[1]), int(x + offset[0])] == 10:
                    self.center_line_direction = -1
                    self.previous_center_line = (int(x + offset[0]), int(y + offset[1]))
                    return int(x + offset[0]), int(y + offset[1])
                
        if debug: print("Didnt find center line, using previous")

        self.center_line_direction = 0
        return self.previous_center_line
    
    def get_center_line_3_dir(self, x, y, dir):
        direction = 9999
        error = 999
        offset = None
        dir %= 360
        for i in range(len(offsets)):
            if self.track[int(y + offsets[i][1]), int(x + offsets[i][0])] == 10:
                if angle_distance(dir, directions[i]) < error or direction > 360:
                    error = angle_distance(dir, directions[i])
                    direction = directions[i]
                    offset = offsets[i]
            if (error < 60): break
        return direction, offset
    
    def center_line_3_input(self, x, y, dir=None):
        distances = travel_distances_centerlines
        if not dir or dir == 9999:
            dir = self.direction
            direction, offset = self.get_center_line_3_dir(x, y, dir)
        else:
            direction = dir
            offset = offsets[directions.tolist().index(direction)]
        if len(offset) == 0:
            return [(self.x, self.y)] * len(distances)
        results = []
        assert(self.track[int(y), int(x)] == 10)

        for i in range(max(distances) + 1):
            if i in distances:
                results.append((int(x), int(y)))
            x += offset[0]
            y += offset[1]
            possible_offsets = [
                potential_offset
                for potential_offset in offsets.tolist()
                if angle_distance(direction, directions[offsets.tolist().index(potential_offset)]) <= 90
            ]
            for potential_offset in possible_offsets:
                if self.track[int(y + potential_offset[1]), int(x + potential_offset[0])] == 10:
                    offset = potential_offset
                    index = np.where((offsets == offset).all(axis=1))[0]
                    direction = directions[index]
        return results
    
    def calculateScore(self):
        current_x = self.start_x
        current_y = self.start_y
        current_dir = self.start_direction
        final_x = self.previous_center_line[0]
        final_y = self.previous_center_line[1]
        seen = 0
        assert(self.track[current_y, current_x] == 10)
        while (current_x, current_y) != (final_x, final_y) and seen < 50000:
            potential_offsets = [offset for offset in offsets if angle_distance(current_dir, np.degrees(np.arctan2(-offset[1], offset[0]))) <= 90 and self.track[current_y + offset[1], current_x + offset[0]] == 10]
            if not potential_offsets: break
            current_offset = random.choice(potential_offsets)
            current_x += current_offset[0]
            current_y += current_offset[1]
            current_dir = np.degrees(np.arctan2(-current_offset[1], current_offset[0]))
            seen += 1
        if (seen > 8000 and len(self.checkpoints_seen) < 1) or seen == 50000:
            return 0
        return min(1, seen / self.calculateMaxPotential())
    
    def calculateMaxPotential(self):
        current_x = self.start_x
        current_y = self.start_y
        current_dir = self.start_direction
        seen = 0
        while not any([self.track[current_y + offset[1], current_x + offset[0]] == 3 for offset in offsets]) and seen < 50000:
            potential_offsets = [offset for offset in offsets if angle_distance(current_dir, np.degrees(np.arctan2(-offset[1], offset[0]))) <= 90 and self.track[current_y + offset[1], current_x + offset[0]] == 10]
            if not potential_offsets: break
            current_offset = random.choice(potential_offsets)
            current_x += current_offset[0]
            current_y += current_offset[1]
            current_dir = np.degrees(np.arctan2(-current_offset[1], current_offset[0]))
            seen += 1
        return max(1, seen)
    
    def validIndex(self, x, y):
        return y < len(self.track) and y >= 0 and x < len(self.track[0]) and x >= 0
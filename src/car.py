import pygame
import numpy as np
import json

from utils import calculate_distance, next_speed, angle_distance, new_brake_speed
from precomputed import sin, cos, offsets, tan, arctan, speed_squared
from settings import *

class Car:
    def __init__(self, track, start_pos, start_dir, track_name=None, speed_pre_calc=True):
        self.track = track
        self.track_name = track_name or False

        with open('./src/config.json', 'r') as json_file:
            config_data = json.load(json_file)
        
        self.drag_coeff = config_data['drag_coeffs'].get(track_name)
        self.ppm = config_data['pixel_per_meter'].get(track_name)

        self.c_length = car_length * self.ppm
        self.c_width = car_width * self.ppm
        
        self.x = start_pos[1]
        self.int_x = int(self.x)
        self.y = start_pos[0]
        self.int_y = int(self.y)
        self.direction = start_dir
        self.int_direction = int(self.direction)

        assert(self.track[self.int_y, self.int_x] == 10)

        self.start_x = self.x
        self.start_y = self.y
        self.start_direction  = self.direction
        self.previous_center_line = (self.x, self.y)
        self.center_line_dist = 0
        self.finish_x = self.x
        self.finish_y = self.y
        self.score, self.laps = 0, 0

        self.front_left, self.front_right, self.back_left, self.back_right = [0, 0], [0, 0], [0, 0], [0, 0]
        self.acceleration, self.brake, self.speed, self.steer = 0, 0, 0, 0
        self.lap_times, self.lap_time = [], 0
        self.center_line_direction = None
        self.previous_points = np.array([None] * len(points_offset))
        self.future_corners = []

        self.checkpoints_seen, self.checkpoints = [], []

        self.seen = 0
        self.max_pot_seen = 0

        self.end_pos = config_data['end_pos'].get(track_name)

        self.UpdateCorners()

        self.speed_pre_calc = None
        if speed_pre_calc:
            self.pre_calc_speed = []
            for i in range(0, 3399, 1):
                self.pre_calc_speed.append(next_speed(i/10))
            self.speed_pre_calc = np.array(self.pre_calc_speed)

        self.died = False
        self.track_max_potential = None

        self.safe_from_end = False
        self.end_check = 0
        self.safe_from_corner = False
        self.corner_check = 0
        
    def Tick(self, game):
        self.ApplyPlayerInputs()
        self.UpdateCar()
        return self.CheckCollisions(game.ticks)

    def CheckCollisions(self, ticks):
        # Keep the two offsets closest to 90 degrees from the car from offsets
        track_val = self.track[self.int_y, self.int_x]
        if len(self.future_corners) > 0:
            corner_x, corner_y, _, _ = self.future_corners[0]
            next_corner_dist = calculate_distance((corner_x, corner_y), (self.x, self.y))
            if next_corner_dist < 20 * self.ppm:
                # Calculate angle to corner, if > 90 then remove it
                angle = np.degrees(-np.arctan2(corner_y - self.y, corner_x - self.x))
                angle_diff = angle_distance(self.direction, angle)
                if angle_diff > 50:
                    self.future_corners.pop(0)

        if track_val == 0:
            self.GetNearestCenterline()
            self.UpdateCorners()
            toCheck = [self.front_left, self.front_right, self.back_left, self.back_right]
            count = 0
            for point in toCheck:
                if self.track[int(point[1]), int(point[0])] == 0: count += 1
                else: break

            if count > 3 and not god:
                self.Kill()
                return False
        elif track_val == 2:
            seen = False
            for checkpoint in self.checkpoints_seen:
                if calculate_distance(checkpoint, (self.x, self.y)) < min_checkpoint_distance:
                    seen = True
            if seen == False:
                self.checkpoints_seen.append((self.x, self.y, ticks))
        if (not self.safe_from_end) or ticks - self.end_check > 10:
            self.end_check = ticks
            self.safe_from_end = calculate_distance((self.x, self.y), self.end_pos) > 75 * self.ppm
            if not self.safe_from_end:
                res = (
                    track_val == 3 or
                    self.track[self.int_y + 3, self.int_x + 3] == 3 or
                    self.track[self.int_y - 3, self.int_x - 3] == 3 or
                    self.track[self.int_y + 3, self.int_x - 3] == 3 or
                    self.track[self.int_y - 3, self.int_x + 3] == 3
                )
                if res:
                    self.GetNearestCenterline()
                    self.lap_time = ticks
                    if len(self.checkpoints_seen) < 1 and angle_distance(self.direction, self.start_direction) > 90: # The car isn't facing the correct direction
                        self.lap_time = 0
                        print("Cheating attempt")
                    self.Kill()
                    return False
        return True
    
    def Kill(self):
        self.died = True
        self.finish_x, self.finish_y = self.GetNearestCenterline()

        self.safe_from_end = False
        self.safe_from_corner = False
        self.end_check = 0
        self.corner_check = 0

        self.x = self.start_x
        self.y = self.start_y
        self.direction = self.start_direction
        self.speed = 0
        self.acceleration = 0
        self.brake = 0
        self.steer = 0

        self.UpdateCorners()

    def Accelerate(self):
        self.acceleration += acceleration_increment
        self.brake -= acceleration_increment * 2
        self.brake = max(0, self.brake)
        self.acceleration = min(1, self.acceleration)

    def Decelerate(self):
        self.brake += brake_increment
        self.acceleration -= brake_increment * 2
        self.acceleration = max(0, self.acceleration)
        self.brake = min(1, self.brake)

    def UpdateSteer(self, value):
        if value > 0:
            self.steer += steer_increment
            self.steer = min(1, self.steer)
        elif value < 0:
            self.steer -= steer_increment
            self.steer = max(-1, self.steer)

    def ApplyPlayerInputs(self):
        keys = pygame.key.get_pressed()
        power, steer, brake = False, False, False
        if keys[pygame.K_w]:
            self.Accelerate()
            power = True
        if keys[pygame.K_s]:
            self.Decelerate()
            brake = True
        if keys[pygame.K_a]:
            self.UpdateSteer(1)
            steer = True
        if keys[pygame.K_d]:
            self.UpdateSteer(-1)
            steer = True
       
        self.CheckForAction(brake, power, steer)
        
    def ApplyAgentInputs(self, action):
        acc, steer = action
        steer_change, brake_change, power_change = False, False, False
        if acc > 0.5:
            self.Accelerate()
            power_change = True
        elif acc <-0.5:
            self.Decelerate()
            brake_change = True
        if steer > 0.5:
            self.UpdateSteer(1)
            steer_change = True
        elif steer < -0.5:
            self.UpdateSteer(-1)
            steer_change = True
        self.CheckForAction(brake_change, power_change, steer_change)
    
    def CheckForAction(self, brake, power, steer):
        if brake == False:
            if(self.brake > 0):
                self.brake -= delta_t * 3
                self.brake = max(0, self.brake)
            elif self.brake < 0:
                self.brake += delta_t * 3
                self.brake = min(0, self.brake)
        if power == False:
            if(self.acceleration > 0):
                self.acceleration -= delta_t * 3
                self.acceleration = max(0, self.acceleration)
            elif self.acceleration < 0:
                self.acceleration += delta_t * 3
                self.acceleration = min(0, self.acceleration)
        if steer == False:
            if(self.steer > 0):
                self.steer -= delta_t * 3
                self.steer = max(0, self.steer)
            elif self.steer < 0:
                self.steer += delta_t * 3
                self.steer = min(0, self.steer)
        if self.steer > 1:
            self.steer = 1
        elif self.steer < -1:
            self.steer = -1
        
    def UpdateCar(self):     
        speed_factor = max(1.0 - pow(int(self.speed) / (max_speed), 0.5), 0.1)
        wheel_angle = speed_factor * self.steer * 14

        tan_angle = tan[int((wheel_angle % 180) * angle_resolution_factor)]
        turning_radius = self.c_length / tan_angle if tan_angle != 0 else float('inf')
        arctan_angle = self.c_length / (turning_radius - self.c_width / 2)
        if arctan_angle < 0: wheel_angle = -arctan[int(-arctan_angle*arctan_resolution_factor)]
        else: wheel_angle = arctan[int(arctan_angle*arctan_resolution_factor)]
       
        self.direction += wheel_angle * delta_t * (self.speed + 20) * turn_coeff

        # - Car speed
        self.speed += (next_speed(self.speed, self.speed_pre_calc) - self.speed) * self.acceleration
        if self.brake > 0: self.speed += (new_brake_speed(self.speed) - self.speed) * self.brake

        drag_force = 0.5 * (drag_coeff) * (reference_area) * speed_squared[int(self.speed * 10)]
        drag_acceleration = drag_force * delta_t / car_mass

        self.speed -= drag_acceleration * ((1-self.acceleration) * (1-self.brake) + abs(self.steer) * 1/4)
        
        displacement = (self.speed / 3.6) * self.ppm
        self.x += displacement * cos[(self.int_direction) * angle_resolution_factor] * delta_t
        self.y -= displacement * sin[(self.int_direction) * angle_resolution_factor] * delta_t
        
        self.direction %= 360
        self.int_direction = int(self.direction)
        self.int_x = int(self.x)
        self.int_y = int(self.y)
        

    def UpdateCorners(self):
        half_small_side = car_width * self.ppm / 2
        half_big_side = car_length * self.ppm / 2
        angle = arctan[int(half_small_side / half_big_side * arctan_resolution_factor)]
        cos_1 = cos[(int(self.direction + angle - 90)% 360) * angle_resolution_factor]
        sin_1 = sin[(int(self.direction + angle - 90)% 360) * angle_resolution_factor]

        self.front_left[0] = self.x - half_small_side * cos_1 - half_big_side * sin_1
        self.front_left[1] = self.y - half_big_side * cos_1 + half_small_side * sin_1
        self.front_right[0] = self.x + half_small_side * cos_1 - half_big_side * sin_1
        self.front_right[1] = self.y - half_big_side * cos_1 - half_small_side * sin_1

        self.back_left[0] = self.x - half_small_side * cos_1 + half_big_side * sin_1
        self.back_left[1] = self.y + half_big_side * cos_1 + half_small_side * sin_1
        self.back_right[0] = self.x + half_small_side * cos_1 + half_big_side * sin_1
        self.back_right[1] = self.y + half_big_side * cos_1 - half_small_side * sin_1

    def CenterlinePosition(self, x, y):
        if self.track[y, x] == 10:
            return x, y
        return None
    
    def UpdatePreCalc(self):
        self.int_x = int(self.x)
        self.int_y = int(self.y)
        self.int_direction = int(self.direction)

    def GetNearestCenterline(self, game=None):
        if self.died: return self.previous_center_line
        normalised_x, normalised_y = self.int_x, self.int_y
        if self.track[normalised_y, normalised_x] == 10:
            self.previous_center_line = (normalised_x, normalised_y)
            self.center_line_direction = 0
            self.center_line_dist = 0
            return (normalised_x, normalised_y)
        remaining_directions = [1, -1, 0, 2]
        for i in range(int(max_center_line_distance * self.ppm + 10)):
            for direction in remaining_directions:
                angle = ((self.int_direction + direction * 90) % 360) * angle_resolution_factor
                x = int(self.x + i * cos[angle] * 2)
                y = int(self.y - i * sin[angle] * 2)
                if x < 1 or y < 1 or x >= len(self.track[0]) - 1 or y >= len(self.track) - 1: continue
                if self.track[y, x] == 10:
                    self.previous_center_line = (x, y)
                    self.center_line_direction = direction
                    self.center_line_dist = i
                    return (x, y)
                for offset in offsets:
                    new_x, new_y = x + offset[0], y + offset[1]
                    
                    if self.track[new_y, new_x] == 10:
                        self.previous_center_line = (new_x, new_y)
                        self.center_line_direction = direction
                        self.center_line_dist = i
                        return new_x, new_y
        
        if game is not None and game.debug:
            print(f"Didnt find center line, using previous. Ticknum: {game.ticks}")
            game.issues.value += 1

        self.center_line_direction = 0
        return self.previous_center_line
    
    def CalculateScore(self, max_potential=None):
        current_x = self.start_x
        current_y = self.start_y
        current_dir = self.start_direction
        final_x = self.finish_x
        final_y = self.finish_y
        seen = 0
        if self.lap_time > 0:
            return 1
        if self.lap_time == -1:
            return 0
        if max_potential is None:
            max_potential = self.CalculateMaxPotential()
        calculated_dirs = {}
        for offset in offsets:
            calculated_dirs[offset[0] + 1000 * offset[1]] = np.degrees(np.arctan2(-offset[1], offset[0]))
            
        while ((final_x, final_y) != (current_x, current_y)) and seen < 50000:
            potential_offsets = [offset for offset in offsets if angle_distance(current_dir, calculated_dirs[offset[0] + 1000 * offset[1]]) <= 90 and self.track[current_y + offset[1], current_x + offset[0]] == 10]
            if not potential_offsets: break
            current_offset = potential_offsets[0]
            current_x += current_offset[0]
            current_y += current_offset[1]
            current_dir = calculated_dirs[current_offset[0] + 1000 * current_offset[1]]
            seen += 1

        self.seen = seen
        return min(1, seen / max_potential)
    
    def CalculateMaxPotential(self, current_dir=None):
        current_x = self.start_x
        current_y = self.start_y
        if current_dir is None:
            current_dir = self.start_direction
        seen = 0
        calculated_dirs = {}
        for offset in offsets:
            calculated_dirs[offset[0] + 1000 * offset[1]] = np.degrees(np.arctan2(-offset[1], offset[0]))
        assert(self.track[current_y, current_x] == 10)
        while (not any([self.track[current_y + offset[1], current_x + offset[0]] == 3 for offset in offsets])) and seen < 50000:
            potential_offsets = [offset for offset in offsets if angle_distance(current_dir, calculated_dirs[offset[0] + 1000 * offset[1]]) <= 90 and self.track[current_y + offset[1], current_x + offset[0]] == 10]
            if potential_offsets == []: 
                raise Exception(f"Car has no potential, seen: {seen}, track: {self.track_name}, x: {current_x}, y: {current_y}, dir: {current_dir}")
            current_offset = potential_offsets[0]
            current_x += current_offset[0]
            current_y += current_offset[1]
            current_dir = calculated_dirs[current_offset[0] + 1000 * current_offset[1]]
            seen += 1
        self.max_pot_seen = seen
        return max(1, seen) 

    def setFutureCorners(self, corners_data):
        corners = [corner[0] for corner in corners_data]
        corners_amplitude = [corner[1] for corner in corners_data]
        ordered_corners = []

        self.UpdatePreCalc()
        self.GetNearestCenterline()

        current_x, current_y = self.previous_center_line
        current_dir = self.direction

        prev_directions = [self.direction] * 10

        while not any([self.track[current_y + offset[1], current_x + offset[0]] == 3 for offset in offsets]):
            potential_offsets = [offset for offset in offsets if angle_distance(current_dir, np.degrees(np.arctan2(-offset[1], offset[0]))) <= 90 and self.track[current_y + offset[1], current_x + offset[0]] == 10]
            if len(potential_offsets) == 0:
                print(f"Weird, {ordered_corners}, {current_x}, {current_y}, {current_dir}")
                break
            current_offset = potential_offsets[0]
            current_x += current_offset[0]
            current_y += current_offset[1]
            new_dir = np.degrees(np.arctan2(-current_offset[1], current_offset[0]))
            current_dir = new_dir
            if (current_x, current_y) in corners:
                amplitude = corners_amplitude[corners.index((current_x, current_y))]
                ordered_corners.append((int(current_x), int(current_y), int(new_dir), int(amplitude)))
            prev_directions.append(new_dir)
            prev_directions.pop(0)
        ordered_corners.append((current_x, current_y, int(current_dir), 0))
        ordered_corners.append((0, 0, 0, 0))
        ordered_corners.append((0, 0, 0, 0))
    
        self.future_corners = ordered_corners
        return ordered_corners
    
    def MaxPotential(self, score_dict):
        start_seen = score_dict[int(str(self.start_x) + str(self.start_y))]
        potential_offsets = [offset for offset in offsets if self.track[self.start_y + offset[1], self.start_x + offset[0]] == 10]
        # Take the offset that minimises the angle distance to start_dir
        best_offset = min(potential_offsets, key=lambda offset: angle_distance(self.start_direction, np.degrees(np.arctan2(-offset[1], offset[0]))))
        offset_seen = score_dict[int(str(self.start_x + best_offset[0]) + str(self.start_y + best_offset[1]))]
        # If it is positive, then return the number of keys - start_seen
        if offset_seen > start_seen:
            return len(score_dict.keys()) - start_seen
        else:
            return start_seen
        
    def ScoreCar(self, score_dict, game=None):
        try:
            seen = score_dict[int(str(self.previous_center_line[0]) + str(self.previous_center_line[1]))]
            start_seen = score_dict[int(str(self.start_x) + str(self.start_y))]
            return abs(seen - start_seen)
        except:
            print(f"Error in score, {self.previous_center_line}, {self.start_x}, {self.start_y}")
            self.CalculateScore(1)

            return self.seen

        
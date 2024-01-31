import pygame
import numpy as np
import json

from utils import calculate_distance, next_speed, angle_distance, new_brake_speed
from precomputed import sin, cos, offsets, directions
from settings import *

class Car:
    def __init__(self, track, start_pos, start_dir, track_name=None):
        self.track = track
        self.track_name = track_name or False

        with open('./src/config.json', 'r') as json_file:
            config_data = json.load(json_file)
        self.ppm = config_data['pixel_per_meter'].get(track_name)
        
        self.x = start_pos[1]
        self.y = start_pos[0]
        self.direction = start_dir

        assert(self.track[int(self.y), int(self.x)] == 10)

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
        self.checkpoints_seen, self.checkpoints = [], []
        self.previous_points = np.array([None] * len(points_offset))
        self.future_corners = []

        self.UpdateCorners()

        self.died = False
        self.track_max_potential = None
        
    def Tick(self, game):
        self.ApplyPlayerInputs()
        self.UpdateCar()
        return self.CheckCollisions(game.ticks)

    def CheckCollisions(self, ticks):
        track_val = self.track[int(self.y), int(self.x)]
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
        elif track_val == 3:
            self.GetNearestCenterline()
            self.lap_time = ticks
            if len(self.checkpoints_seen) < 1 and angle_distance(self.direction, self.start_direction) > 90: # The car isn't facing the correct direction
                self.lap_time = 0
            self.Kill()
            return False
        elif track_val == 2:
            seen = False
            for checkpoint in self.checkpoints_seen:
                if calculate_distance(checkpoint, (self.x, self.y)) < min_checkpoint_distance:
                    seen = True
            if seen == False:
                self.checkpoints_seen.append((self.x, self.y, ticks))
        # Get Distance to first corner in future corners
        if len(self.future_corners) > 0:
            corner_x, corner_y, _ = self.future_corners[0]
            next_corner_dist = calculate_distance((corner_x, corner_y), (self.x, self.y))
            if next_corner_dist < 10 * self.ppm:
                self.future_corners.pop(0)
        return True
    
    def Kill(self):
        self.died = True
        self.finish_x, self.finish_y = self.GetNearestCenterline()

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
        power, steer = action
        steer_change, brake_change, power_change = False, False, False
        if power > 0.8:
            self.Accelerate()
            power_change = True
        elif power < -0.8:
            self.Decelerate()
            brake_change = True
        if steer > 0.8:
            self.UpdateSteer(1)
            steer_change = True
        elif steer < -0.8:
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
        if self.steer < -1:
            self.steer = -1
        
    def UpdateCar(self):     
        wheel_angle = self.steer * 14 
        speed_factor = max(1.0 - pow(int(self.speed) / (max_speed), 0.5), 0.1)

        wheel_angle *= speed_factor
        if wheel_angle != 0: turning_radius = car_length * self.ppm / np.tan(np.radians(wheel_angle))
        else: turning_radius = float('inf')  # Ligne droite
        wheel_angle = np.arctan(car_length * self.ppm / (turning_radius - car_width * self.ppm / 2))

        self.direction += wheel_angle * delta_t * (self.speed + 20) * turn_coeff

        # - Car speed
        self.speed += (next_speed(self.speed) - self.speed) * self.acceleration
        if self.brake > 0: self.speed += (new_brake_speed(self.speed) - self.speed) * self.brake

        drag_force = 0.5 * drag_coeff * reference_area * pow(self.speed, 2)
        drag_acceleration = drag_force / car_mass
        self.speed -= drag_acceleration * delta_t * (1-self.acceleration) * (1-self.brake)

        displacement = (self.speed / 3.6) * self.ppm
        self.x += displacement * cos[(int(self.direction) % 360) * 10] * delta_t
        self.y -= displacement * sin[(int(self.direction) % 360) * 10] * delta_t

        self.direction %= 360


    def UpdateCorners(self):

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

    def CalculatePoint(self, offset):
        dx = 1.0
        angle = int(self.direction + 90 + offset) % 360
        sinus = sin[angle * 10]
        cosinus = cos[angle * 10]
        eval_x, eval_y = dx * sinus + self.x, dx * cosinus + self.y

        distance_from_track = 0
        while int(eval_x) < len(self.track[0]) and int(eval_y) < len(self.track) and self.track[int(eval_y), int(eval_x)] == 0 and distance_from_track < 50:
            distance_from_track += 5
            dx += 5.0
            eval_x, eval_y = dx * sinus + self.x, dx * cosinus + self.y
        
        if int(eval_x) >= len(self.track[0]) or int(eval_y) >= len(self.track):
            return np.array([eval_x, eval_y])
        
        if distance_from_track == 50: return (self.x,self.y)

        i = 0
        while self.track[int(eval_y), int(eval_x)] != 0 and i < max_points_distance * self.ppm:
            dx += point_search_jump
            eval_x, eval_y = dx * sinus + self.x, dx * cosinus + self.y
            i += point_search_jump

        jump = point_search_jump / 2.0
        while jump > 1.0:
            if self.track[int(eval_y), int(eval_x)] == 0:
                dx -= jump
            else:
                dx += jump
            eval_x, eval_y = dx * sinus + self.x, dx * cosinus + self.y
            jump /= 2.0

        eval_x, eval_y = dx * sinus + self.x, dx * cosinus + self.y

        return np.array([eval_x, eval_y])
    
    def GetPointsInput(self):
        return 
        self.previous_points = np.array([self.CalculatePoint(offset) for offset in points_offset])
        return self.previous_points
    
    def CenterlinePosition(self, x, y):
        if self.track[y, x] == 10:
            return x, y
        return None

    def GetNearestCenterline(self, game=None):
        if self.died: return self.previous_center_line
        normalised_x, normalised_y = int(self.x), int(self.y)
        if self.track[normalised_y, normalised_x] == 10:
            self.previous_center_line = (normalised_x, normalised_y)
            self.center_line_direction = 0
            self.center_line_dist = 0
            return (normalised_x, normalised_y)
        remaining_directions = [1, -1, 0, 2]
        for i in range(int(max_center_line_distance * self.ppm + 10)):
            for direction in remaining_directions:
                angle = int(self.direction + direction * 90) % 360
                x = int(self.x + i * cos[int(angle * 10)] * 2)
                y = int(self.y - i * sin[int(angle * 10)] * 2)
                if x < 1 or y < 1 or x >= len(self.track[0]) - 1 or y >= len(self.track) - 1: continue
                if self.track[y, x] == 10:
                    self.previous_center_line = (x, y)
                    self.center_line_direction = direction
                    self.center_line_dist = i
                    return (x, y)
                for offset in offsets:
                    new_x, new_y = int(x + offset[0]), int(y + offset[1])
                    
                    if self.track[new_y, new_x] == 10:
                        self.previous_center_line = (new_x, new_y)
                        self.center_line_direction = direction
                        self.center_line_dist = i
                        return new_x, new_y
        
        if game is not None and game.debug:
            print(f"Didnt find center line, using previous. Ticknum: {game.ticks}")

        self.center_line_direction = 0
        return self.previous_center_line
    
    def CalculateNextCenterlineDirection(self, x, y, direction_target):
        direction = 9999
        error = 999
        offset = None
        direction_target %= 360
        
        for offset, direction in zip(offsets, directions):
            x_offset, y_offset = int(x + offset[0]), int(y + offset[1])
            track_value = self.track[y_offset, x_offset]
            
            if track_value == 10:
                current_error = angle_distance(direction_target, direction)
                if current_error < error or direction > 360:
                    error = current_error
                    direction = direction
                    offset = offset
            if error < 60:
                break
        
        return direction, offset
    
    def CalculateScore(self, max_potential=None):
        current_x = self.start_x
        current_y = self.start_y
        current_dir = self.start_direction
        final_x = self.finish_x
        final_y = self.finish_y
        seen = 0
        if self.lap_time > 0:
            return 1
        if max_potential is None:
            max_potential = self.CalculateMaxPotential(current_dir)
        while (final_x, final_y) != (current_x, current_y) and seen < 50000:
            potential_offsets = [offset for offset in offsets if angle_distance(current_dir, np.degrees(np.arctan2(-offset[1], offset[0]))) <= 90 and self.track[current_y + offset[1], current_x + offset[0]] == 10]
            if not potential_offsets: break
            current_offset = potential_offsets[0]
            current_x += current_offset[0]
            current_y += current_offset[1]
            current_dir = np.degrees(np.arctan2(-current_offset[1], current_offset[0]))
            seen += 1
        if (seen == 50000 or (seen > 8000 and len(self.checkpoints_seen) < 1)):
            return 0

        return min(1, seen / max_potential)
    
    def CalculateMaxPotential(self, current_dir=None):
        current_x = self.start_x
        current_y = self.start_y
        if current_dir is None:
            current_dir = self.start_direction
        seen = 0
        while (not any([self.track[current_y + offset[1], current_x + offset[0]] == 3 for offset in offsets])) and seen < 50000:
            potential_offsets = [offset for offset in offsets if angle_distance(current_dir, np.degrees(np.arctan2(-offset[1], offset[0]))) <= 90 and self.track[current_y + offset[1], current_x + offset[0]] == 10]
            if not potential_offsets: 
                if debug: print(f"No potential offsets, track: {self.track_name}")
                break
            current_offset = potential_offsets[0]
            current_x += current_offset[0]
            current_y += current_offset[1]
            current_dir = np.degrees(np.arctan2(-current_offset[1], current_offset[0]))
            seen += 1
        return max(1, seen)
    
    def get_point_further(self, x, y, direction, max_dist, track):
        current_x, current_y = x, y
        current_dir = direction
        for i in range(max_dist):
            valid_offsets = [offset for offset in offsets if track[current_y + offset[1], current_x + offset[0]] == 10 and angle_distance(current_dir, np.degrees(np.arctan2(-offset[1], offset[0]))) <= 60]
            if not valid_offsets:
                print("Lost")
                break
            chosen_offset = valid_offsets[0]
            chosen_angle = np.degrees(np.arctan2(-chosen_offset[1], chosen_offset[0]))
            
            # Update current position and direction
            current_dir = chosen_angle
            current_x += chosen_offset[0]
            current_y += chosen_offset[1]

        return current_x, current_y, current_dir 

    def setFutureCorners(self, corners):
        ordered_corners = []
        
        current_x, current_y = self.previous_center_line
        current_dir = self.direction

        prev_directions = [self.direction] * 10

        while not any([self.track[current_y + offset[1], current_x + offset[0]] == 3 for offset in offsets]):
            potential_offsets = [offset for offset in offsets if angle_distance(current_dir, np.degrees(np.arctan2(-offset[1], offset[0]))) <= 90 and self.track[current_y + offset[1], current_x + offset[0]] == 10]
            if not potential_offsets: break
            current_offset = potential_offsets[0]
            current_x += current_offset[0]
            current_y += current_offset[1]
            new_dir = np.degrees(np.arctan2(-current_offset[1], current_offset[0]))
            current_dir = new_dir
            if (current_x, current_y) in corners:
                ordered_corners.append((current_x, current_y, new_dir))
            prev_directions.append(new_dir)
            prev_directions.pop(0)
        self.future_corners = ordered_corners
        return ordered_corners

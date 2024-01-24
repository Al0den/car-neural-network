import numpy as np
import matplotlib.pyplot as plt
import threading
from matplotlib.animation import FuncAnimation

from car import Car
from utils import calculate_distance, GetCenterlineInputs, angle_range_180, calculate_distance_vectorized
from settings import *

max_corner_distance = 300

class Agent:
    def __init__(self, options, track, start_pos, start_dir, track_name=None):
        self.car = Car(track, start_pos, start_dir, track_name)
        self.InitializeNetwork(options)
        self.evolution = ["r"]
        self.mutation_rates = ["-"]
        self.state = []
        self.action = [0, 0]

        self.attempted = True
        self.mutation_strengh = mutation_strenght
        self.last_update = 0

    def CalculateState(self, game, ticks, calculated_points=None):
        center_line_x, center_line_y = self.car.GetNearestCenterline(game)

        if calculated_points is None:
            input_data = []
            for offset in points_offset:
                input_data += [int(self.car.x), int(self.car.y), int(self.car.direction + 90 + offset) % 360, game.track_index[self.car.track_name], int(self.car.ppm * 1000)]
            input_data = np.array(input_data)
            calculated_points = game.Metal.getPointsOffset(input_data.flatten().astype(np.int32))

        distance_to_center_line = min(1, max(-1, calculate_distance((center_line_x, center_line_y), (self.car.x, self.car.y)) * self.car.center_line_direction / (self.car.ppm * max_center_line_distance)))
        state = [self.car.speed/360, self.car.acceleration, self.car.brake, self.car.steer, distance_to_center_line]
        if self.car.future_corners == []:
            next_corner_distance = 0
            relative_angle = 0
            left_or_right = 0
        else:
            next_corner_x, next_corner_y, next_corner_dir = self.car.future_corners[0]
            next_corner_distance = min(1, calculate_distance((next_corner_x, next_corner_y), (self.car.x, self.car.y)) / (self.car.ppm * max_corner_distance))
            relative_angle = angle_range_180(self.car.direction - next_corner_dir)
            left_or_right = 1 if relative_angle > 0 else -1 if relative_angle < 0 else 0
        state += [next_corner_distance, abs(relative_angle) / 180, left_or_right]

        if len(self.car.future_corners) > 1:
            next_corner_x, next_corner_y, next_corner_dir = self.car.future_corners[1]
            prev_corner_x, prev_corner_y, _ = self.car.future_corners[0]
            next_corner_distance = min(1, calculate_distance((next_corner_x, next_corner_y), (prev_corner_x, prev_corner_y)) / (self.car.ppm * max_corner_distance))
            relative_angle = angle_range_180(self.car.direction - next_corner_dir)
            left_or_right = 1 if relative_angle > 0 else -1 if relative_angle < 0 else 0
        else:
            next_corner_distance = 0
            relative_angle = 0
            left_or_right = 0
        state += [next_corner_distance, abs(relative_angle) / 180, left_or_right]

        calculated_points_input = np.minimum(1, np.maximum(-1, calculated_points / (self.car.ppm * max_points_distance)))
        state += calculated_points_input.tolist()
    
        if game.debug: 
            for inp in state:
                if np.abs(inp) > 1: 
                    print("One of the inputs is out of bounds, input num: " + str(np.where(state == inp)[0]))

        return state
    
    def Tick(self, ticks, game, calculated_points=None):
        if self.car.died == True: return
        self.mutation_strengh = game.mutation_strength

        state = self.CalculateState(game, ticks, calculated_points)
        
        power, steer = self.CalculateNextAction(state)
        self.car.ApplyAgentInputs([power, steer])
        self.car.UpdateCar()
        self.car.CheckCollisions(ticks)

        self.state = state
        self.action = [power, steer]

        if ticks > safety_ticks and self.car.speed < min_speed:
            self.car.Kill()
        if ticks > max_ticks_before_kill:
            self.car.Kill()
        return state

    def InitializeNetwork(self, options):
        network = []
        state_size = options['state_space_size']
        hidden_layer_size = options['hidden_layer_size']
        action_size = options['action_space_size']
        num_hidden_layers = options['num_hidden_layers']

        layer_sizes = [state_size]

        if num_hidden_layers == 0:
            layer_sizes.append(action_size)
        else:
            step = ((state_size - action_size) * first_layer_size_coeff) / (num_hidden_layers + 1)
            for i in range(num_hidden_layers):
                current_hidden_size = (state_size * first_layer_size_coeff) - step * (i + 1)
                layer_sizes.append(int(current_hidden_size))
            layer_sizes.insert(1, hidden_layer_size) 
            layer_sizes.append(action_size)

        for i in range(len(layer_sizes) - 1):
            network.append(np.random.uniform(-1, 1, size=(layer_sizes[i], layer_sizes[i + 1])))

        self.network = network

    def Mutate(self, rate):
        for layer in self.network:
            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    int = np.random.uniform(0, 1)
                    if int < rate:
                        layer[i][j] += np.random.uniform(-mutation_strenght, mutation_strenght)
                        layer[i][j] = min(1, max(-1, layer[i][j]))
    
    def CalculateNextAction(self, state):  
        current_layer_output = state
        for layer_weights in self.network:
            current_layer_output =  activation_function(np.dot(current_layer_output, layer_weights))
        return current_layer_output
    
    def AgentDistance(self, agent2):
        # Calculate the euclidian distance, wihout using calculate distance
        difference = 0
        for i in range(len(self.network)):
            for j in range(len(self.network[i])):
                for k in range(len(self.network[i][j])):
                    difference += abs(self.network[i][j][k] - agent2.network[i][j][k])
        return difference
    
    def SetAgent(self, start, track=None, track_name=None):
        self.car.x = start[0][1]
        self.car.y = start[0][0]
        self.car.start_x = start[0][1]
        self.car.start_y = start[0][0]
        self.car.direction = start[1]
        self.car.start_direction = start[1]
        self.car.speed = 0
        self.car.acceleration = 0
        self.car.brake = 0
        self.car.steer = 0
        self.car.died = False
        self.car.previous_points = []
        self.car.lap_time = 0
        self.lap_time = 0
        self.score = 0
        self.laps = 0
        self.car.laps = 0
        self.car.lap_times = 0
        self.car.score = 0
        self.car.lap_time = 0
        if track is not None:
            self.car.track = track
            self.car.track_name = track_name

    


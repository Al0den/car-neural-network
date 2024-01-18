import numpy as np
import matplotlib.pyplot as plt
import threading
from matplotlib.animation import FuncAnimation

from car import Car
from utils import calculate_distance, GetCenterlineInputs, angle_range_180, calculate_distance_vectorized
from settings import *

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
        points = GetCenterlineInputs(game, self.car)

        if calculated_points is None:
            input_data = []
            for offset in points_offset:
                input_data += [int(self.car.x), int(self.car.y), int(self.car.direction + 90 + offset) % 360, game.track_index[self.car.track_name]]
            calculated_points = game.getPointsOffset(np.array(input_data).flatten().astype(np.int32))
            calculated_points = calculated_points.reshape((len(points_offset), 2))

        distance_to_center_line = calculate_distance((center_line_x, center_line_y), (self.car.x, self.car.y)) * self.car.center_line_direction / (self.car.ppm * max_center_line_distance)
        state = [self.car.speed/360, self.car.acceleration, self.car.brake, self.car.steer, distance_to_center_line] + points
        car_pos = np.array([self.car.x, self.car.y])
        distances = calculate_distance_vectorized(calculated_points, car_pos)
        normalized_distances = np.maximum(-1, np.minimum(1, distances / (max_points_distance * self.car.ppm) * 1.2))

        state += normalized_distances.tolist()

        if debug: 
            for inp in state:
                if abs(inp) > 1: print("One of the inputs is out of bounds, input num: " + str(state.index(inp)))

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
        if track is not None:
            self.car.track = track
            self.car.track_name = track_name

    


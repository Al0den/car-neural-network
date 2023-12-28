import numpy as np

from car import Car
from utils import calculate_distance, get_centerline_points, angle_range_180
from settings import *

class Agent:
    def __init__(self, options, track, start_pos, start_dir, track_name=None):
        self.car = Car(track, start_pos, start_dir, track_name)
        self.initialize_network(options)
        self.evolution = ["r"]
        self.mutation_rates = ["-"]
        self.state = []
        self.action = []

        self.attempted = True
        self.mutation_strengh = mutation_strenght
        self.last_update = 0
    
    def tick(self, ticks, game, player=0):
        if self.car.died == True: return
        self.mutation_strengh = game.mutation_strength

        center_line_x, center_line_y = self.car.get_centerline()
        points = get_centerline_points(game, self.car)
        self.car.getPoints()
        distance_to_center_line = calculate_distance((center_line_x, center_line_y), (self.car.x, self.car.y)) * self.car.center_line_direction / (self.car.ppm * max_center_line_distance)
        state = [self.car.speed/360, self.car.acceleration, self.car.brake, self.car.steer, distance_to_center_line, self.action[0], self.action[1]]
        
        prev_angle = self.car.direction
        for i in range(len(points)):
            if i == 0: continue
            angle = np.degrees(np.arctan2(-points[i][1] + points[i-1][1], points[i][0] - points[i-1][0]))
            angle_diff = angle_range_180(angle - prev_angle)
            prev_angle = angle
            state.append(min(1, angle_diff / 70))
        
        for point in self.car.previous_points:
            state.append(min(1, calculate_distance(point, (self.car.x, self.car.y)) / (max_points_distance * self.car.ppm)* 1.2))

        if debug: 
            for inp in state:
                if abs(inp) > 1: print("One of the inputs is iout of bounds")
        
        power, steer = self.get_action(state)
        self.car.applyAgentInputs([power, steer])
        self.car.updateCar()
        self.car.checkCollisions(ticks, player)

        self.state = state
        self.action = [power, steer]

        if ticks > 50 and self.car.speed < min_speed:
            self.car.kill()
            self.car.died = True
        if ticks > max_ticks_before_kill:
            self.car.kill()
        return state

    def initialize_network(self, options):
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

    def mutate(self, rate):
        for layer in self.network:
            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    int = np.random.uniform(0, 1)
                    if int < rate:
                        layer[i][j] += np.random.uniform(-mutation_strenght, mutation_strenght)
                        layer[i][j] = min(1, max(-1, layer[i][j]))
    
    def get_action(self, state):  
        current_layer_output = state
        for layer_weights in self.network:
            current_layer_output =  activation_function(np.dot(current_layer_output, layer_weights))
        return current_layer_output
    
    


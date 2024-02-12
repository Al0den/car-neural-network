import numpy as np

from car import Car
from utils import calculate_distance, angle_range_180
from settings import *

max_corner_distance = 800

class Agent:
    def __init__(self, options, track, start_pos, start_dir, track_name=None):
        self.car = Car(track, start_pos, start_dir, track_name)
        self.InitializeNetwork(options)
        self.evolution = ["r"]
        self.mutation_rates = ["-"]
        self.state = np.array([0.0] * state_space_size)
        self.action = np.array([0, 0])

        self.attempted = True
        self.mutation_strengh = mutation_strenght
        self.last_update = 0

    def ProcessCorner(self, corner):
        corner_x, corner_y, corner_dir, corner_ampl = corner
        distance = min(1, calculate_distance((corner_x, corner_y), (self.car.x, self.car.y)) / (self.car.ppm * max_corner_distance))
        relative_angle = angle_range_180(self.car.direction - corner_dir)
        left_or_right = 1 if relative_angle > 0 else -1 if relative_angle < 0 else 0
        return [distance, left_or_right, min(1, corner_ampl/100)]

    def CalculateState(self, game, calculated_points=None):
        if calculated_points is None:
            game.Metal.inVectorBuffer[0:5] = [self.car.int_x, self.car.int_y, self.car.int_direction, game.track_index[self.car.track_name], int(self.car.ppm * 1000)]
            game.Metal.getPointsOffset(len(points_offset))
            calculated_points = game.Metal.outVectorBuffer[:len(points_offset)]
        
        self.state[0:4] = [self.car.speed/360, self.car.acceleration, self.car.brake, self.car.steer]
        if len(self.car.future_corners) >= 2: 
            self.state[4:7] = self.ProcessCorner(self.car.future_corners[0])
            self.state[7:10] = self.ProcessCorner(self.car.future_corners[1])
        elif len(self.car.future_corners) == 1:
            self.state[4:7] = self.ProcessCorner(self.car.future_corners[0])
            self.state[7:10] = [0, 0, 0]
        else:
            self.state[4:10] = [0, 0, 0, 0, 0, 0]

        self.state[10:] = np.minimum(1, calculated_points / (self.car.ppm * max_points_distance))

        return self.state
    
    def Tick(self, ticks, game, calculated_points=None, updateCar=True):
        if self.car.died == True: return

        self.CalculateState(game, calculated_points)
        
        power, steer = self.CalculateNextAction(self.state)

        self.car.ApplyAgentInputs([power, steer])

        self.car.UpdateCar()
        self.car.CheckCollisions(ticks)

        self.action[0:2] = [power, steer]

        if ticks > safety_ticks and self.car.speed < min_speed:
            self.car.Kill()
        if ticks > max_ticks_before_kill:
            self.car.Kill()

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
            current_layer_output = np.tanh(np.dot(current_layer_output, layer_weights))
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
        self.car.UpdateCorners()

    


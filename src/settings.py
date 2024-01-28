import numpy as np

points_offset = [0, 4, -4,  15, -15, 30, -30, 45, -45, 60, -60, 90, -90]

# - Neural network settings
center_line_input = True
state_space_size = len(points_offset) + 10
first_layer_size_coeff = 1.3
num_hidden_layers = 3
action_space_size = 2
map_tries = 20

real_starts_num = 6
assert(real_starts_num <= map_tries)

max_points_distance = 200 
max_center_line_distance = 20

first_layer_size = int(state_space_size * first_layer_size_coeff)

# - Parameters
point_search_jump = 10.0
perft_ticks = 5
perft_duration = 20
center_line_hash_size = 100000
specialised_training_multiple = 1
compile_shaders = False

# - Game settings
god = False
debug = False
random_start_position = True
delta_t = 1/60
pre_load = True
min_checkpoint_distance = 200
max_time_on_checkpoint = 200
safety_ticks = 60
base_game_speed = 60

# - Simulation settings
car_length = 5.23
car_width = 1.8
max_speed = 340
turn_coeff = 4.5
drag_coeff = 0.881
reference_area = 1.7
lift_coeff = -1.247
car_mass = 733
gravity = 9.81
brake_increment = 1/15
acceleration_increment = 1/15
steer_increment = 1/15

# - Evolution settings
mutation_rates = [0.01, 0.1, 0.2, 0.3]

only_mutate_rate = 0.6
cross_over_rate = 0.35
mutate_cross_over_rate = 0.7
simple_cross_over_rate = 0.3
random_agent_rate = 0.05

previous_ratio = 0.05
no_lap_value = 1000000
score_multiplier = 10000

mutation_strenght = 0.2

max_ticks_before_kill = 20000
min_speed = 5
base_score_per_checkpoint = 0

agent_selection_coeff = 7

# - Render Options
default_width = 1220
default_height = 780
slider_width = 200
slider_height = 20
slider_padding = 10  # Padding from the right and bottom edges


max_int = 2147483647


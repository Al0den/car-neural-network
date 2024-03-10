import os
from settings import *
import readline
from colorama import Fore, init

init()

def complete_track(text, state):
    tracks = [file[:-4] for file in os.listdir("data/tracks") if file.endswith(".png") and not file.endswith("_surface.png")]
    options = [track for track in tracks if track.startswith(text)]
    return options[state] if state < len(options) else None

def load_options():
    environment_options = {
        'num_agents': 1,
        'state_space_size': state_space_size,
        'action_space_size': action_space_size,
        'hidden_layer_size': first_layer_size,
        'num_hidden_layers': num_hidden_layers,
        'generation_to_load': 0,
        'track_name': None
    }
    
    game_options = {
        'track_name': 'singapoure',
        'player': 0, 
        'screen_width': default_width,
        'screen_height': default_height,
        'environment': environment_options,
        'display': False,
        'generation_to_load': 0,
        'visual': True,
        'cores': 1,
        'track_data': None,
        'track_name': None,
        'mutation_strength': mutation_strenght,
    }

    game_options['player'] = int(input("Player (0 = player, 1 = Train, 2 = Continue Train, 3 = Specific Train, 4 = Specific Display, 5 = Continuous Display, 6 = Race, 7 = Draw Line, 8 = Show Multiple Agents, 9 = Generate All Lines, 10 = Benchmark):"))
    if game_options['player'] == 0:
        game_options['display'] = True
    elif game_options['player'] == 1:
        game_options['environment']['num_agents'] = int(input("Number of agents: "))
        game_options['cores'] = int(input("Cores: "))
    elif game_options['player'] == 2:
        game_options['cores'] = int(input("Cores: "))
    elif game_options['player'] == 3:
        game_options['cores'] = int(input("Cores: "))
        game_options['mutation_strength'] = float(input("Mutation strength: "))
    elif game_options['player'] == 4:
        game_options['environment']['generation_to_load'] = int(input("Generation to load - Only number: "))
        game_options['generation_to_load'] = game_options['environment']['generation_to_load']
        game_options['display'] = True
    elif game_options['player'] == 5:
        game_options['display'] = True
        game_options['cores'] = 1
    elif game_options['player'] == 6:
        game_options['display'] = True
        game_options['environment']['num_agents'] = 2
        game_options['cores'] = 1
    elif game_options['player'] == 7:
        game_options['cores'] = 1
        game_options['environment']['generation_to_load'] = int(input("Generation to load - Only number: "))
        game_options['generation_to_load'] = game_options['environment']['generation_to_load']
    elif game_options['player'] == 8:
        game_options['display'] = True
        game_options['environment']['num_agents'] = int(input("Number of agents: "))
        game_options['cores'] = 1
        game_options['s_or_g'] = input("Specific (s) or global (g) trained agents?:")
        if game_options['s_or_g'] == "g":
            game_options['gap'] = int(input("Gap between agents?: "))
    elif game_options['player'] == 9:
        game_options['display'] == True
        game_options['environment']['num_agents'] = int(input("Number of agents: "))
        game_options['cores'] = 1
    elif game_options['player'] == 10:
        game_options['cores'] = 1
        game_options['environment']['num_agents'] = 10
    
    if game_options['player'] in [0, 3, 4, 6, 7, 8, 9]:
        readline.parse_and_bind("tab: complete")
        readline.set_completer(complete_track)

        print("Available tracks:")
        for file in os.listdir("data/tracks"):
            if file.endswith(".png") and not file.endswith("_surface.png"):
                track_name = file[:-4]
                npy_file_path = os.path.join("data/tracks", f"{track_name}.npy")
                per_track_file_path = os.path.join("data/per_track", track_name)
                if os.path.exists(per_track_file_path):
                    print(f" - {Fore.BLUE}{track_name}{Fore.RESET}")
                elif os.path.exists(npy_file_path):
                    print(f" - {Fore.GREEN}{track_name}{Fore.RESET}")
                else:
                    print(f" - {Fore.RED}{track_name}{Fore.RESET}")

        game_options['track_name'] = input("Track: ")
    

    os.system("clear")
    print(f" * Config: Cores - {str(game_options['cores'])}, Agents - {str(game_options['environment']['num_agents'])}, Track - {game_options['track_name']}, Mode - {number_to_player(game_options['player'])}") 
    return game_options

def number_to_player(num):
    if num == 0: return "Human"
    elif num == 1: return "AI Training"
    elif num == 2: return "AI Continue Training"
    elif num == 3: return "AI Specific Map Testing"
    elif num == 4: return "AI Specific Map Display"
    elif num == 5: return "AI Display"
    elif num == 6: return "AI vs Human"
    elif num == 7: return "Draw optimal line"
    elif num == 8: return "Show multiple agents"
    elif num == 9: return "Generate all lines"
    elif num == 10: return "Performance Test"
    
import pygame
import numpy as np
import os

from utils import calculate_distance, get_centerline_points, copy_car
from settings import debug
from agent import Agent
from car import Car
from precomputed import pixel_per_meter

class Render:
    def __init__(self, screen, game_options, game):
        self.screen = screen
        self.game_options = game_options
        self.red = (255, 0, 0)
        self.gray = (100, 100, 100)
        self.orange = (168, 155, 50)
        self.car_width = 5
        self.car_length = 10
        car_image = pygame.image.load("./data/f1.png")
        self.car_image = pygame.transform.scale(car_image, (600, 228))
        self.debug_val = debug
        self.visible_track = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        self.visible_track.set_alpha(None)
        self.tracks = game.tracks
        self.surfaces = {}
        for track_name in self.tracks:
            self.surfaces[track_name] = self.load_track_surface(track_name)

    def load_track_surface(self, track_name):
        if os.path.isfile("data/tracks/" + track_name + "_surface.png"):
            return pygame.image.load("data/tracks/" + track_name + "_surface.png")
        else:
            print(f" - Error loading track surface for track {track_name}")
    
    def draw_track(self, track_matrix, position, screen, game):
        x_offset = max(0, min(track_matrix.shape[1] - screen.get_width(), position[0] - screen.get_width() // 2))
        y_offset = max(0, min(track_matrix.shape[0] - screen.get_height(), position[1] - screen.get_height() // 2))

        if self.visible_track.get_size() != screen.get_size():
            self.visible_track = pygame.Surface(screen.get_size())

        self.visible_track.blit(self.surfaces[game.track_name], (0, 0), (x_offset, y_offset, screen.get_width(), screen.get_height()))
        screen.blit(self.visible_track, (0, 0))

    def info(self, car, game):
        lines = [
            f"Direction: {car.direction:.1f} degrees",
            f"Speed: {car.speed:.1f} km/h",
            f"Score: {int(car.score)}",
            f"Track: {game.track_name}" 
        ]

        y_offset = 10
        font = pygame.font.SysFont("Arial", 18)

        for line in lines:
            text_surface = font.render(line, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += text_surface.get_height() + 5

    def debug(self, game):
        if game.player == 0: self.generate_agent_debug(game)
        else: self.use_agent_debug(game)
    def draw_car(self, car, camera_x, camera_y, color, track, displayCar=False):
        if displayCar:
            front_x = car.x + self.car_length * np.cos(np.radians(car.direction % 360)) - camera_x
            front_y = car.y - self.car_length * np.sin(np.radians(car.direction % 360)) - camera_y

            dot_radius = self.car_width / 3 
            dot_x = front_x + dot_radius * np.cos(np.radians(car.direction))
            dot_y = front_y - dot_radius * np.sin(np.radians(car.direction))
        
            pygame.draw.circle(self.screen, color, (int(dot_x), int(dot_y)), int(dot_radius))
        
            front_left_in_track = track[int(car.front_left[1])][int(car.front_left[0])] != 0
            front_right_in_track = track[int(car.front_right[1])][int(car.front_right[0])] != 0
            back_left_in_track = track[int(car.back_left[1])][int(car.back_left[0])] != 0
            back_right_in_track = track[int(car.back_right[1])][int(car.back_right[0])] != 0
            if front_left_in_track != 0: front_left_color = color 
            else: front_left_color = (255, 0, 0)
            if front_right_in_track: front_right_color = color
            else: front_right_color = (255, 0, 0)
            if back_left_in_track: back_left_color = color
            else: back_left_color = (255, 0, 0)
            if back_right_in_track: back_right_color = color
            else: back_right_color = (255, 0, 0)
            pygame.draw.circle(self.screen, front_left_color, (car.front_left[0] - camera_x, car.front_left[1] - camera_y), 2)
            pygame.draw.circle(self.screen, front_right_color, (car.front_right[0] - camera_x, car.front_right[1]- camera_y), 2)
            pygame.draw.circle(self.screen, back_left_color, (car.back_left[0] - camera_x, car.back_left[1]- camera_y), 2)
            pygame.draw.circle(self.screen, back_right_color, (car.back_right[0] - camera_x, car.back_right[1]- camera_y), 2)
        else:
            car_image = pygame.transform.rotate(self.car_image, car.direction)
            car_center_x = car.x - camera_x 
            car_center_y = car.y - camera_y  

            scaling_factor = pixel_per_meter[car.track_name] /  (2.3 * 30)

            scaled_car_image = pygame.transform.scale(car_image, (
                int(car_image.get_width() * scaling_factor),
                int(car_image.get_height() * scaling_factor)
            ))

            blit_x = car_center_x - scaled_car_image.get_width() // 2
            blit_y = car_center_y - scaled_car_image.get_height() // 2

            self.screen.blit(scaled_car_image, (blit_x, blit_y))
            
    def draw_acceleration(self, acceleration, braking, x, y, width, height):
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, width, height))
        if acceleration > 0:
            accel_fill_color = (0, int(255 * acceleration), 0)
        else:
            accel_fill_color = (int(255 * abs(acceleration)), 0, 0)

        accel_fill_height = abs(acceleration) * height / 2

        accel_fill_rect = pygame.Rect(x, y + height / 2 - accel_fill_height, width, accel_fill_height)
        pygame.draw.rect(self.screen, accel_fill_color, accel_fill_rect)

        if braking > 0:
            brake_fill_color = (int(255 * braking), 0, 0)
        else:
            brake_fill_color = (0, int(255 * abs(braking)), 0)

        brake_fill_height = abs(braking) * height / 2

        brake_fill_rect = pygame.Rect(x, y + height / 2, width, brake_fill_height)
        pygame.draw.rect(self.screen, brake_fill_color, brake_fill_rect)
    def draw_steering(self, steering, x, y, width, height):
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, width, height))

        if steering < 0:
            fill_color = (0, int(255 * abs(steering)), 0)
            fill_width = abs(steering) * width / 2
            fill_x = x + width / 2
        else:
            fill_color = (int(255 * abs(steering)), 0, 0)
            fill_width = abs(steering) * width / 2
            fill_x = x + width / 2 - fill_width

        fill_rect = pygame.Rect(fill_x, y, fill_width, height)
        pygame.draw.rect(self.screen, fill_color, fill_rect)

        pygame.draw.line(self.screen, (255, 255, 255), (x + width / 2, y), (x + width / 2, y + height))

    def draw_points(self, car, camera_x, camera_y, prev_points=None):
        if prev_points == None:
            car.getPoints()
            points = car.previous_points
        else:
            points = prev_points
        if (points[0] == None): return
        for point in points:
            target_x = point[0]
            target_y = point[1]
            if calculate_distance((car.x, car.y),(target_x, target_y)) > 10:
                pygame.draw.circle(self.screen, (0, 0, 255), (int(target_x - camera_x), int(target_y - camera_y)), 2)

    def draw_lines(self, game, camera_x, camera_y, car):
        car.get_centerline()
        points = get_centerline_points(game, car)
        first = True
        for point in points:
            if first:
                pygame.draw.circle(self.screen, (255, 0, 0), (int(point[0]) - camera_x, int(point[1]) - camera_y), 3)
                first = False
            else:
                pygame.draw.circle(self.screen, (255, 128, 0), (int(point[0]) - camera_x, int(point[1]) - camera_y), 3)

    def use_agent_debug(self, game):
        tps = game.clock.get_fps()
        actions = [f"{action:0.2f}" for action in game.environment.agents[0].action]
        state = [f"{state:0.2f}" for state in game.environment.agents[0].state]

        state_lines = [state[i:i + 5] for i in range(0, len(state), 5)]

        lines = [
            f"Generation: {game.environment.generation}",
            f"Best lap: {game.environment.previous_best_lap}",
            f"Best score: {game.environment.previous_best_score}",
            f"tps: {tps:.1f}",
            f"Action: [{', '.join(actions)}]",
            "State:"
        ]

        for line in state_lines:
            lines.append(f"  [{', '.join(line)}]")
        lines[-1] = lines[-1][:-1]

        y_offset = 10
        font = pygame.font.SysFont("Arial", 18)
        w, _ = pygame.display.get_surface().get_size()

        for line in lines:
            text_surface = font.render(line, True, (0, 0, 0))
            self.screen.blit(text_surface, (w * 0.75, y_offset))
            y_offset += text_surface.get_height() + 5
    def generate_agent_debug(self, game):
        new_car = Car(game.car.track, game.start_pos, game.start_dir, game.track_name)
        new_car = copy_car(game.car, new_car)
        new_car.died = False
        agent = Agent(game.options['environment'], game.track, game.start_pos, game.start_dir, game.track_name)

        agent.car = new_car
        agent.tick(0, game)
        agent.car.get_centerline()

        actions = [f"{action:0.2f}" for action in agent.action]
        state = [f"{state:0.2f}" for state in agent.state]

        state_lines = [state[i:i + 5] for i in range(0, len(state), 5)]

        lines = [
            f"Action: [{', '.join(actions)}]",
            "State:"
        ]

        for line in state_lines:
            lines.append(f"  [{', '.join(line)}]")
        lines[-1] = lines[-1][:-1]

        y_offset = 10
        font = pygame.font.SysFont("Arial", 18)
        w, _ = pygame.display.get_surface().get_size()

        for line in lines:
            text_surface = font.render(line, True, (0, 0, 0))
            self.screen.blit(text_surface, (w * 0.75, y_offset))
            y_offset += text_surface.get_height() + 5

    def RenderFrame(self, game):
        centered_car = None
        if game.player == 0: centered_car = game.car 
        else: centered_car = game.environment.agents[0].car
        camera_x = centered_car.x - game.screen.get_width() // 2
        camera_y = centered_car.y - game.screen.get_height() // 2

        if game.visual:
            self.draw_track(game.track, (centered_car.x, centered_car.y), self.screen, game)
            if not game.debug: self.draw_car(centered_car, camera_x, camera_y, (0,255,255), game.track, False)
            else: self.draw_car(centered_car, camera_x, camera_y, (0,255,255), game.track, True)
        else:
            self.screen.fill((255, 255, 255))
            self.draw_car(centered_car, camera_x, camera_y, (0,255,255), game.track, True)
        self.draw_acceleration(centered_car.acceleration, centered_car.brake, 10,300, 20, 100)
        self.draw_steering(centered_car.steer, 40, 300, 100, 20)
        if game.player != 0:
            for agent in game.environment.agents:
                if agent.car == centered_car: continue
                self.draw_car(agent.car, camera_x, camera_y, (0, 255, 0), game.track)
        self.info(centered_car, game)

        if game.debug or not game.visual:
            self.debug(game)
            self.draw_points(centered_car, camera_x, camera_y)
            self.draw_lines(game, camera_x, camera_y, centered_car)

        pygame.display.update()

        


        

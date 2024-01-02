import pygame
import numpy as np
import os

from utils import calculate_distance, GetCenterlineInputs, copy_car
from settings import *
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
        
        self.tracks = game.tracks
        self.surfaces = {}
        for track_name in self.tracks:
            self.surfaces[track_name] = self.load_track_surface(track_name)

        self.slider_value = 0.0  # Initial value of the slider
        self.slider_x = self.screen.get_width() - slider_width - slider_padding  # X position of the slider
        self.slider_y = self.screen.get_height() - slider_height - slider_padding  # Y position of the slider
        self.slider_dragging = False  # Flag to track if the slider is being dragged
        self.zoom_factor = 6 / pixel_per_meter[game.track_name]
        self.zoomed_width = int(screen.get_width() / self.zoom_factor)
        self.zoomed_height = int(screen.get_height() / self.zoom_factor)
        self.visible_track = pygame.Surface((self.zoomed_width, self.zoomed_height), pygame.SRCALPHA)
        self.visible_track.set_alpha(None)

        self.zoom_offset = 0


    def load_track_surface(self, track_name):
        if os.path.isfile("data/tracks/" + track_name + "_surface.png"):
            return pygame.image.load("data/tracks/" + track_name + "_surface.png")
        else:
            print(f" - Error loading track surface for track {track_name}")
    
    def draw_track(self, position, screen, game):
        zoom_factor = self.zoom_factor  # Get the zoom factor

        center_x = position[0]
        center_y = position[1]

        zoomed_width = int(screen.get_width() / zoom_factor)
        zoomed_height = int(screen.get_height() / zoom_factor)

        x_offset = center_x - zoomed_width // 2
        y_offset = center_y - zoomed_height // 2

        zoomed_track_surface = pygame.Surface((zoomed_width, zoomed_height), pygame.SRCALPHA)
        zoomed_track_surface.set_alpha(None)
        zoomed_track_surface.fill((0, 100, 0))

        source_rect = pygame.Rect(x_offset, y_offset, zoomed_width, zoomed_height)
        destination_rect = pygame.Rect(0, 0, zoomed_width, zoomed_height)

        zoomed_track_surface.blit(self.surfaces[game.track_name], destination_rect, source_rect)
        zoomed_visible_track = pygame.transform.scale(zoomed_track_surface, (screen.get_width(), screen.get_height()))

        screen.blit(zoomed_visible_track, (0, 0))

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
    def DrawCar(self, car, camera_x, camera_y, color, track, displayCar=False):
        if displayCar:
            corners = [car.front_left, car.front_right, car.back_right, car.back_left]
            for i in range(len(corners)):
                car_distance = calculate_distance((car.x, car.y), (corners[i][0], corners[i][1]))
                new_dist = car_distance * self.zoom_factor
                angle = np.arctan2(corners[i][1] - car.y, corners[i][0] - car.x)

                new_x = car.x + new_dist * np.cos(angle)
                new_y = car.y + new_dist * np.sin(angle)
                adjusted_x = int(new_x - camera_x)
                adjusted_y = int(new_y - camera_y)

                if track[int(corners[i][1]), int(corners[i][0])] == 0:
                    pygame.draw.circle(self.screen, (255, 0, 0), (adjusted_x, adjusted_y), 4)
                else:
                    pygame.draw.circle(self.screen, color, (adjusted_x, adjusted_y), 4)
            # Draw lines between the corners, doing similar math to get adjusted coordinates
            for i in range(len(corners)):
                car_distance = calculate_distance((car.x, car.y), (corners[i][0], corners[i][1]))
                new_dist = car_distance * self.zoom_factor
                angle = np.arctan2(corners[i][1] - car.y, corners[i][0] - car.x)

                new_x = car.x + new_dist * np.cos(angle)
                new_y = car.y + new_dist * np.sin(angle)
                adjusted_x = int(new_x - camera_x)
                adjusted_y = int(new_y - camera_y)

                if i == len(corners) - 1:
                    next_corner = corners[0]
                else:
                    next_corner = corners[i + 1]

                next_car_distance = calculate_distance((car.x, car.y), (next_corner[0], next_corner[1]))
                next_new_dist = next_car_distance * self.zoom_factor
                next_angle = np.arctan2(next_corner[1] - car.y, next_corner[0] - car.x)

                next_new_x = car.x + next_new_dist * np.cos(next_angle)
                next_new_y = car.y + next_new_dist * np.sin(next_angle)
                next_adjusted_x = int(next_new_x - camera_x)
                next_adjusted_y = int(next_new_y - camera_y)

                pygame.draw.line(self.screen, color, (adjusted_x, adjusted_y), (next_adjusted_x, next_adjusted_y), 2)
                
        else:
            car_image = pygame.transform.rotate(self.car_image, car.direction)
            car_center_x = car.x - camera_x
            car_center_y = car.y - camera_y

            scaling_factor = pixel_per_meter[car.track_name] /  (2.3 * 30) * self.zoom_factor

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

    def DrawPointsInput(self, car, camera_x, camera_y, prev_points=None):
        if prev_points == None:
            car.GetPointsInput()
            points = car.previous_points
        else:
            points = prev_points
        if (points[0] == None): return
        for point in points:
            target_x = point[0]
            target_y = point[1]
            distance = calculate_distance((car.x, car.y), (target_x, target_y))
            new_dist = distance * self.zoom_factor

            angle = np.arctan2(target_y - car.y, target_x - car.x)

            new_x = car.x + new_dist * np.cos(angle)
            new_y = car.y + new_dist * np.sin(angle)
            pygame.draw.line(self.screen, (0, 0, 255), (int(car.x - camera_x), int(car.y - camera_y)), (int(new_x - camera_x), int(new_y - camera_y)))
            pygame.draw.circle(self.screen, (0, 0, 255), (int(new_x - camera_x), int(new_y - camera_y)), 3)

    def DrawCenterlineInputs(self, game, camera_x, camera_y, car):
        car.GetNearestCenterline(game)
        points = GetCenterlineInputs(game, car)
        first = True

        prev_x = car.x
        prev_y = car.y

        for point in points:
            target_x = point[0]
            target_y = point[1]
            distance = calculate_distance((car.x, car.y), (target_x, target_y))
            new_dist = distance * self.zoom_factor

            angle = np.arctan2(target_y - car.y, target_x - car.x)

            new_x = car.x + new_dist * np.cos(angle)
            new_y = car.y + new_dist * np.sin(angle)

            if first:
                pygame.draw.circle(self.screen, (255, 0, 0), (new_x - camera_x, new_y - camera_y), 3)
                first = False
            else:
                pygame.draw.line(self.screen, (255, 0, 0), ((new_x - camera_x), (new_y - camera_y)), ((prev_x - camera_x), (prev_y - camera_y)))
                pygame.draw.circle(self.screen, (255, 128, 0), ((new_x - camera_x), (new_y - camera_y)), 3)
            prev_x = new_x
            prev_y = new_y

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
        agent.Tick(0, game)
        agent.car.GetNearestCenterline(game)

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
    def handle_slider(self, game):
        pygame.draw.rect(self.screen, (200, 200, 200), (self.slider_x, self.slider_y, slider_width, slider_height))
        slider_pos = self.slider_x + self.slider_value * slider_width
        pygame.draw.rect(self.screen, (0, 0, 0), (slider_pos - 5, self.slider_y - 5, 10, slider_height + 10))
        if game.player == 0:
            car = game.car
        else:
            car = game.environment.agents[0].car
        # Handle mouse events
        self.slider_value = car.direction / 360
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if self.slider_x <= mouse_x <= self.slider_x + slider_width and self.slider_y <= mouse_y <= self.slider_y + slider_height:
                    self.slider_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.slider_dragging = False
            elif event.type == pygame.MOUSEMOTION and self.slider_dragging:
                mouse_x, _ = pygame.mouse.get_pos()
                self.slider_value = max(0, min(1, (mouse_x - self.slider_x) / slider_width))  # Convert to range -1 to 1
                if game.player in [0, 4, 5]:
                    car.direction = self.slider_value * 360
    def update_zoom_offset(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_EQUALS]:
            self.zoom_offset += 0.03
        if keys[pygame.K_MINUS]:
            self.zoom_offset -= 0.03

    def RenderFrame(self, game):
        self.update_zoom_offset()
        centered_car = None
        if game.player == 0: centered_car = game.car 
        else: centered_car = game.environment.agents[0].car
        self.zoom_factor = (4 / pixel_per_meter[centered_car.track_name]) + self.zoom_offset
        camera_x = centered_car.x - game.screen.get_width() // 2
        camera_y = centered_car.y - game.screen.get_height() // 2
        x_cos = np.cos(np.radians(centered_car.direction))
        y_sin = np.sin(np.radians(centered_car.direction))
        steer_var = centered_car.steer * 4 * centered_car.speed / 180

        acceleration_var = centered_car.acceleration * 3 * (centered_car.speed / 140 + 0.2)
        brake_var = centered_car.brake * 3 * (centered_car.speed / 140 + 0.2)

        total_power_var = acceleration_var - brake_var

        offset_x, offset_y = steer_var * y_sin + total_power_var * x_cos, steer_var * x_cos - total_power_var * y_sin

        if game.visual:
            self.draw_track((centered_car.x, centered_car.y), self.screen, game)
            if not game.debug: self.DrawCar(centered_car, camera_x + offset_x, camera_y + offset_y, (0,255,255), game.track, False)
            else: self.DrawCar(centered_car, camera_x + offset_x, camera_y + offset_y, (0,255,255), game.track, True)
        else:
            self.screen.fill((255, 255, 255))
            self.DrawCar(centered_car, camera_x + offset_x, camera_y + offset_y, (0,255,255), game.track, True)
        self.draw_acceleration(centered_car.acceleration, centered_car.brake, 10,300, 20, 100)
        self.draw_steering(centered_car.steer, 40, 300, 100, 20)
        if game.player != 0:
            for agent in game.environment.agents:
                if agent.car == centered_car: continue
                self.DrawCar(agent.car, camera_x + offset_x, camera_y + offset_y, (0, 255, 0), game.track, True)
        self.info(centered_car, game)

        if game.debug or not game.visual:
            self.debug(game)
            self.DrawPointsInput(centered_car, camera_x + offset_x, camera_y + offset_y)
            self.DrawCenterlineInputs(game, camera_x + offset_x, camera_y + offset_y, centered_car)

        self.handle_slider(game)
        pygame.display.update()

        


        

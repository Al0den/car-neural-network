import pygame
import numpy as np
import os

import matplotlib.pyplot as plt

from utils import calculate_distance, copy_car, angle_range_180
from settings import *
from agent import Agent
from precomputed import cos, sin

from multiprocessing import Manager, Process


class Render:
    def __init__(self, screen, game_options, game, noRender=False):
        if noRender:
            return
        if game.player in [4, 5]:
            self.shared_data = Manager().list()
            self.graph_process = Process(target=self.draw_perm)
            self.graph_process.start()

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

        self.game = game
        
        self.tracks = game.tracks
        self.surfaces = {}
        for track_name in self.tracks:
            self.surfaces[track_name] = self.load_track_surface(track_name)

        self.slider_value = 0.0  # Initial value of the slider
        self.slider_x = self.screen.get_width() - slider_width - slider_padding  # X position of the slider
        self.slider_y = self.screen.get_height() - slider_height - slider_padding  # Y position of the slider
        self.slider_dragging = False  # Flag to track if the slider is being dragged
        self.zoom_factor = 6 / game.config['pixel_per_meter'].get(game.track_name)
        self.pixel_per_meter = game.config['pixel_per_meter']
        self.zoomed_width = int(screen.get_width() / self.zoom_factor)
        self.zoomed_height = int(screen.get_height() / self.zoom_factor)
        self.visible_track = pygame.Surface((self.zoomed_width, self.zoomed_height), pygame.SRCALPHA)
        self.visible_track.set_alpha(None)

        self.agent = None

        self.zoom_offset = 0

        self.prev_score = None


    def load_track_surface(self, track_name):
        if os.path.isfile("data/tracks/" + track_name + "_surface_path.png"):
            return pygame.image.load("data/tracks/" + track_name + "_surface_path.png")
        elif os.path.isfile("data/tracks/" + track_name + "_surface.png"):
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
    def DrawCar(self, car, camera_x, camera_y, color, track, displayCar=False, carNum=0):
        if car.died == True: return
        screen_center_X = self.screen.get_width() // 2
        screen_center_Y = self.screen.get_height() // 2

        car_center_x = car.x - camera_x
        car_center_y = car.y - camera_y

        distance_to_center_x = car_center_x - screen_center_X
        distance_to_center_y = car_center_y - screen_center_Y

        new_dist_car = calculate_distance((car_center_x, car_center_y), (screen_center_X, screen_center_Y)) * self.zoom_factor
        new_angle_car = np.arctan2(distance_to_center_y, distance_to_center_x)

        new_car_x = screen_center_X + new_dist_car * np.cos(new_angle_car)
        new_car_y = screen_center_Y + new_dist_car * np.sin(new_angle_car)

        if displayCar:
            corners = [car.front_left, car.front_right, car.back_right, car.back_left]
            car.UpdateCorners()
            c_data = []
            for i in range(len(corners)):
                corner_x = corners[i][0] - camera_x
                corner_y = corners[i][1] - camera_y
 
                distance_to_center_x = corner_x - screen_center_X
                distance_to_center_y = corner_y - screen_center_Y

                car_distance = calculate_distance((screen_center_X, screen_center_Y), (corner_x, corner_y))
                new_dist = car_distance * self.zoom_factor
                angle = int(np.degrees(np.arctan2(distance_to_center_y, distance_to_center_x)) * angle_resolution_factor)

                new_x = screen_center_X + new_dist * cos[angle]
                new_y = screen_center_Y + new_dist * sin[angle]

                c_data.append([new_x, new_y])

                if track[int(corners[i][1]), int(corners[i][0])] == 0:
                    pygame.draw.circle(self.screen, (255, 0, 0), (new_x, new_y), 4)
                else:
                    pygame.draw.circle(self.screen, color, (new_x, new_y), 4)
                    
            for i in range(len(corners)):
                c1_data = c_data[i]
                if i == len(corners) - 1: c2_data = c_data[0]
                else: c2_data = c_data[i+1]

                pygame.draw.line(self.screen, color, c1_data, c2_data, 1)

            font = pygame.font.Font(None, 22)
            if carNum > 0: text = font.render(str(abs(carNum)), True, (0, 0, 0))
            else: text = font.render(str(abs(carNum)), True, (255, 0, 0))
            text_rect = text.get_rect(center=(int(new_car_x), int(new_car_y) - 20))
            self.screen.blit(text, text_rect)
        else:
            car_image = pygame.transform.rotate(self.car_image, car.direction)

            scaling_factor = self.pixel_per_meter.get(car.track_name) / (2.3 * 30) * self.zoom_factor
            car_image = pygame.transform.scale(car_image, (int(car_image.get_width() * scaling_factor), int(car_image.get_height() * scaling_factor)))
            self.screen.blit(car_image, (int(new_car_x - car_image.get_width() // 2), int(new_car_y - car_image.get_height() // 2)))
            
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

    def DrawPointsInput(self, car, camera_x, camera_y, game, prev_points=None):
        if game.player not in [4]:
            game.Metal.inVectorBuffer[0:5] = [int(car.x), int(car.y), int(car.direction), game.track_index[car.track_name], int(1000 * car.ppm)]
            game.Metal.getPointsOffset(len(points_offset))
        points_distance = game.Metal.outVectorBuffer[:len(points_offset)]
        points_distance = [point * car.ppm * max_points_distance for point in points_distance]
        points = []
        for i, distance in enumerate(points_distance):
            distance = distance 
            angle_offset = points_offset[i]
            angle = (car.direction + 90 + angle_offset) % 360
            x = car.x + distance * np.sin(np.radians(angle))
            y = car.y + distance * np.cos(np.radians(angle))
            points.append((x, y))
        center_line_x, center_line_y = car.GetNearestCenterline(game)
        # Add to points
        points = np.vstack((points, np.array([center_line_x, center_line_y])))

        for point in points:
            target_x = point[0]
            target_y = point[1]
            distance = calculate_distance((car.x, car.y), (target_x, target_y))
            new_dist = distance * self.zoom_factor

            angle = np.arctan2(target_y - car.y, target_x - car.x)

            new_x = car.x + new_dist * np.cos(angle)
            new_y = car.y + new_dist * np.sin(angle)

            if point[0] == center_line_x and point[1] == center_line_y:
                return  # Ne pas dessiner le point sur la ligne centrale. (Pk il est la deja?)
            else:
                color = (0, 0, 255)
                line_start = int(car.x - camera_x), int(car.y - camera_y)
                line_end = int(new_x - camera_x), int(new_y - camera_y)

                pygame.draw.line(self.screen, color, line_start, line_end)
                pygame.draw.circle(self.screen, color, line_end, 3)

    def DrawLineToNextCorner(self, camera_x, camera_y, car):
        if car.future_corners is None or len(car.future_corners) == 0:
            return

        next_corner = car.future_corners[0]
        next_corner_pos = next_corner[0], next_corner[1]

        car_pos = car.x, car.y
        relative_pos_x, relative_pos_y = next_corner[0] - car.x, next_corner[1] - car.y

        distance_to_corner = calculate_distance(car_pos, next_corner_pos)
        angle_to_corner = np.degrees(np.arctan2(relative_pos_y, relative_pos_x))
        distance_to_corner = distance_to_corner * self.zoom_factor

        angle_rad = np.radians(angle_to_corner)
        new_x = car.x + distance_to_corner * np.cos(angle_rad)
        new_y = car.y + distance_to_corner * np.sin(angle_rad)

        line_start = int(car.x - camera_x), int(car.y - camera_y)
        line_end = int(new_x - camera_x), int(new_y - camera_y)
        if angle_range_180(next_corner[2] - car.direction) > 0:
            color = (255, 0, 0)
            pygame.draw.line(self.screen, color, line_start, line_end, 2)
        else:
            color = (0, 255, 0)
            pygame.draw.line(self.screen, color, line_start, line_end, 2)
        if len(car.future_corners) == 1:
            return
        next_corner = car.future_corners[1]
        distance_to_corner = calculate_distance((car.x, car.y), (next_corner[0], next_corner[1]))
        angle_to_corner = np.degrees(np.arctan2(next_corner[1] - car.y, next_corner[0] - car.x))
        distance_to_corner = distance_to_corner * self.zoom_factor
        new_x = car.x + distance_to_corner * np.cos(np.radians(angle_to_corner))
        new_y = car.y + distance_to_corner * np.sin(np.radians(angle_to_corner))
        if angle_range_180(next_corner[2] - car.future_corners[0][2]) > 0:
            pygame.draw.line(self.screen, (180, 0, 0), (car.x - camera_x, car.y - camera_y), (new_x - camera_x, new_y - camera_y), 2)
        else:
            pygame.draw.line(self.screen, (0, 180, 0), (car.x - camera_x, car.y - camera_y), (new_x - camera_x, new_y - camera_y), 2)
    def use_agent_debug(self, game):
        tps = game.clock.get_fps()
        actions = [f"{action:0.2f}" for action in game.environment.agents[0].action]
        state = [f"{state:0.2f}" for state in game.environment.agents[0].state]
       
        state_lines = [state[i:i + 5] for i in range(0, len(state), 5)]

        lines = [
            f"Generation: {game.environment.generation}",
            f"Current Score: {game.environment.agents[0].car.score}",
            f"Best lap: {game.environment.previous_best_lap}",
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
        if not self.agent or game.track_name != self.agent.car.track_name:
            self.agent = agent = Agent(game.options['environment'], game.track, game.start_pos, game.start_dir, game.track_name)
        agent = self.agent
        new_car = copy_car(game.car, agent.car)
        new_car.died = False

        agent.car = new_car
        agent.car.future_corners = game.car.future_corners
        agent.Tick(game.ticks, game)
        agent.car.GetNearestCenterline(game)

        actions = [f"{action:0.2f}" for action in agent.action]
        state = [f"{state:0.2f}" for state in agent.state]

        state_lines = [state[i:i + 5] for i in range(0, len(state), 5)]
        if len(agent.car.future_corners) == 0:
            next_corner_dir = 0
            next_corner_ampl = 0
            next_corner_dif = 0
            corner_x = agent.car.x
            corner_y = agent.car.y
        else:
            corner_x, corner_y, next_corner_dir, next_corner_ampl = agent.car.future_corners[0]
            next_corner_dif = angle_range_180(next_corner_dir - agent.car.direction)
            next_corner_dir = "Left" if next_corner_dif > 0 else "Right" if next_corner_dif < 0 else "F"

        lines = [
            f"Action: [{', '.join(actions)}]",
            f"tps: {game.clock.get_fps():.1f}",
            f"Next corner Amplitude: {next_corner_ampl:.1f}",
            f"Next corner Direction: {next_corner_dir}",
            f"Next corner Distance: {calculate_distance((agent.car.x, agent.car.y), (corner_x, corner_y)):.1f}",
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
                game.running.value = False
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
        if game.player == 0:
            centered_car = game.car
        else:
            centered_car = game.environment.agents[0].car
        self.zoom_factor = (8 / self.pixel_per_meter.get(centered_car.track_name)) + self.zoom_offset

        camera_x = centered_car.x - game.screen.get_width() // 2
        camera_y = centered_car.y - game.screen.get_height() // 2
        x_cos = np.cos(np.radians(centered_car.direction))
        y_sin = np.sin(np.radians(centered_car.direction))
        steer_var = centered_car.steer * 4 * centered_car.speed / 180 / 8

        acceleration_var = centered_car.acceleration * 3 * (centered_car.speed / 140 + 0.2) / 8
        brake_var = centered_car.brake * 3 * (centered_car.speed / 140 + 0.2) / 8

        total_power_var = acceleration_var - brake_var

        offset_x, offset_y = steer_var * y_sin + total_power_var * x_cos, steer_var * x_cos - total_power_var * y_sin
        if game.visual:
            self.draw_track((centered_car.x, centered_car.y), self.screen, game)
        else:
            self.screen.fill((255, 255, 255))

        self.draw_acceleration(centered_car.acceleration, centered_car.brake, 10,300, 20, 100)
        self.draw_steering(centered_car.steer, 40, 300, 100, 20)

        self.info(centered_car, game)

        if game.debug or not game.visual:
            self.debug(game)
            self.DrawPointsInput(centered_car, camera_x + offset_x, camera_y + offset_y, game)
            self.DrawLineToNextCorner(camera_x + offset_x, camera_y + offset_y, centered_car)

        if game.visual:
            if game.player != 0:
                for i, agent in enumerate(game.environment.agents):
                    if agent.car == centered_car:
                        continue
                    self.DrawCar(agent.car, camera_x + offset_x, camera_y + offset_y, (0, 255, 0), game.track, game.debug, game.car_numbers[i])
            self.DrawCar(centered_car, camera_x + offset_x, camera_y + offset_y, (0,0,0), game.track, game.debug, -game.car_numbers[0])
        else:
            if game.player != 0:
                for i, agent in enumerate(game.environment.agents):
                    if agent.car == centered_car:
                        continue
                    self.DrawCar(agent.car, camera_x + offset_x, camera_y + offset_y, (0, 255, 0), game.track, True, game.car_numbers[i])
            self.DrawCar(centered_car, camera_x + offset_x, camera_y + offset_y, (0,255,255), game.track, True, -game.car_numbers[0])

        self.handle_slider(game)
        pygame.display.update()

    def initDataGraph(self):
        return
        # Run draw_perm in separate process

    def draw_perm(self):
        self.fig, self.axs = plt.subplots(1, 3, figsize=(8, 3))
        self.axs[0].set_xlabel('Time (s)')
        self.axs[0].set_ylabel('Speed (km/h)')
        self.axs[1].set_xlabel('Time (s)')
        self.axs[1].set_ylabel('Throttle')
        self.axs[2].set_xlabel('Time (s)')
        self.axs[2].set_ylabel('Steer')

        # Draw the initial empty lines for each subplot
        self.speed_line, = self.axs[0].plot([], [], 'b-')
        self.throttle_line, = self.axs[1].plot([], [], 'g-', label='Throttle')
        self.brake_line, = self.axs[1].plot([], [], 'r-', label='Brake')
        self.steer_line, = self.axs[2].plot([], [], 'y-')
        print("test")

        plt.show(block=False)

        local_speed = []
        local_throttle = []
        local_brake = []
        local_steer = []
        local_timestamps = []

        tot_data_num = 0

        while True:
            data = self.shared_data[:]
            if data == []:
                continue

            if len(data) < tot_data_num:
                local_speed = []
                local_throttle = []
                local_brake = []
                local_steer = []
                local_timestamps = []
                tot_data_num = 0
                continue

            while len(local_brake) > 300:
                # Pop first element for all local
                local_speed.pop(0)
                local_throttle.pop(0)
                local_brake.pop(0)
                local_steer.pop(0)
                local_timestamps.pop(0)

            # Add anything that wasnt added to local data
            for i in range(tot_data_num, len(data)):
                local_speed.append(data[i][0])
                local_brake.append(data[i][1])
                local_throttle.append(data[i][2])
                local_steer.append(data[i][3])
                local_timestamps.append(data[i][4])

            tot_data_num = len(data)

            self.speed_line.set_data(local_timestamps, local_speed)
            self.throttle_line.set_data(local_timestamps, local_throttle)
            self.brake_line.set_data(local_timestamps, local_brake)
            self.steer_line.set_data(local_timestamps, local_steer)

            # Adjust limits if necessary
            self.axs[0].set_xlim(min(local_timestamps), max(local_timestamps)+0.1)
            self.axs[1].set_xlim(min(local_timestamps), max(local_timestamps)+0.1)
            self.axs[2].set_xlim(min(local_timestamps), max(local_timestamps)+0.1)

            self.axs[0].set_ylim(0, 360)
            self.axs[1].set_ylim(-1.1, 1.1)
            self.axs[2].set_ylim(-1.1, 1.1)

            plt.pause(0.001)

            plt.draw()

    def ClearData(self):
        self.shared_data[:] = []
        return

    def GraphData(self, speed, brake, throttle, steer, timestamp):
        self.shared_data.append([speed, brake, throttle, steer, timestamp])
        return

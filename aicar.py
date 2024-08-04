import math
import sys

import neat
import pygame

WIDTH = 1920
HEIGHT = 1080
CAR_X = 60
CAR_Y = 60
BORDER = (255, 255, 255, 255)

current_generation = 0

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)


class Car:
    def __init__(self):
        # create car module
        self.corners = None
        self.car = pygame.image.load('car.png').convert()
        self.car = pygame.transform.scale(self.car, (CAR_X, CAR_Y))
        self.rotated_car = self.car

        # set up car parameters
        self.position = [830, 920]
        self.angle = 0
        self.speed = 0
        self.set_speed = False

        self.car_center = [self.position[0] + (CAR_X // 2), self.position[1] + (CAR_Y // 2)]

        self.radars = []
        self.radars_to_draw = []

        self.alive = True

        self.distance = 0
        self.time = 0

    def draw(self, screen):
        """Draw a car on map"""
        screen.blit(self.rotated_car, self.position)
        # self.draw_radars(screen)

    def draw_radars(self, screen):
        """Optional, but draws radar lines"""
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.car_center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, ride_map):
        """Checks if car has collided with border of a riding map"""
        self.alive = True
        for point in self.corners:
            if ride_map.get_at((int(point[0]), int(point[1]))) == BORDER:
                self.alive = False
                break

    def check_radars(self, degree, ride_map):
        """Calculates distance to the border for each radar"""
        length = 0
        angle_in_radians = math.radians(360 - (self.angle + degree))
        x = int((self.car_center[0] + math.cos(angle_in_radians) * length))
        y = int((self.car_center[1] + math.sin(angle_in_radians) * length))
        while length < 300:
            x = int((self.car_center[0] + math.cos(angle_in_radians) * length))
            y = int((self.car_center[1] + math.sin(angle_in_radians) * length))
            if ride_map.get_at((x, y)) == BORDER:
                break
            length += 1

        dist = int(math.sqrt(math.pow(x - self.car_center[0], 2) + math.pow(y - self.car_center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, ride_map):
        """Changes car position, speed, and radars"""
        if not self.set_speed:
            self.speed = 20
            self.set_speed = True

        self.rotated_car = self.rotate_center(self.car, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        self.distance += self.speed
        self.time += 1

        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], HEIGHT - 120)

        self.car_center = [int(self.position[0]) + CAR_X / 2, int(self.position[1]) + CAR_Y / 2]

        length = 0.5 * CAR_X
        left_top = [self.car_center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.car_center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.car_center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.car_center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.car_center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.car_center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.car_center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.car_center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(ride_map)
        self.radars.clear()

        for degree in range(-90, 120, 45):
            self.check_radars(degree, ride_map)

    def is_alive(self):
        """Check if the car is still alive"""
        return self.alive

    def retrieve_radar_data(self):
        """Retrieves radar distances for neural network"""
        radars = self.radars
        length_of_radars = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            length_of_radars[i] = int(radar[1] / 30)
        return length_of_radars

    def get_reward(self):
        """Calculates the reward based on the distance traveled"""
        return self.distance / (CAR_X / 2)

    @staticmethod
    def rotate_center(image, angle):
        """Rotates the car"""
        image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        return rotated_image


def run_ai(genomes, config_file):
    pygame.init()
    ride_map = pygame.image.load('map2.png').convert()
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 25)
    alive_font = pygame.font.SysFont("Arial", 20)

    neural_networks = []
    cars = []

    # Create neural networks for all cars (individual genomes)
    for i, genome in genomes:
        neural_network = neat.nn.FeedForwardNetwork.create(genome, config_file)
        neural_networks.append(neural_network)
        genome.fitness = 0
        cars.append(Car())

    global current_generation
    current_generation += 1

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

        for i, car in enumerate(cars):
            output = neural_networks[i].activate(car.retrieve_radar_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10
            elif choice == 1:
                car.angle -= 10
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2
                else:
                    car.speed += 2

        is_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                is_alive += 1
                car.update(ride_map)
                genomes[i][1].fitness += car.get_reward()

        if is_alive == 0:
            break

        # Draw the background
        SCREEN.blit(ride_map, (0, 0))

        # Draw cars that are alive
        for car in cars:
            if car.is_alive():
                car.draw(SCREEN)

        # Display generations and text info
        text = generation_font.render("Generation: " + str(current_generation), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        SCREEN.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(is_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        SCREEN.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         './config.txt')

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(run_ai, 1000)

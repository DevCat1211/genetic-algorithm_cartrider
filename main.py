import pygame
import numpy as np
from numba import *
import tkinter as tk
from tkinter import messagebox

pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 720
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
big_font = pygame.font.SysFont("arial", 30, bold=True)
small_font = pygame.font.SysFont("Segoe UI Symbol", 20)
credit_font = pygame.font.SysFont("malgungothic", 15)
SAVE_FILE_PATH = 'save/save_file.txt'

NEURAL_NETWORK_SIZE = [[8,8]]+[[9,8]]*5+[[9,5]]
POPULATION = 100
DOMINANT_GENE_SURVIVE_RATE = 0.6
MUTATION_RATE = 0.01
SURVIVAL_RATE = 0.3

MAX_SPEED = 10
CAR_WIDTH, CAR_HEIGHT = 30, 60
MAX_SENSOR_DISTANCE = 800
car_image = pygame.image.load("images/car.png")
original_map_lines = [[18,16.3,14,17],[14,17,13,16],[13,16,11,18],[11,18,10,17],[10,17,5,17],[5,17,5,19],[5,19,4,20],[4,20,-5,20],[-5,20,-5,19],[-5,19,-4,17],[-4,17,-9,14],[-9,14,-9,11],[-9,11,6,11],[6,11,6,9],[6,9,1,9],[1,9,-1,7],[-1,7,-1,-1],
    [-1,-1,1,-1],[1,-1,1,7],[1,7,6,7],[6,7,8,9],[8,9,8,11],[8,11,6,13],[6,13,-7,13],[-7,13,-2,16],[-2,16,-2,18],[-2,18,-4,19],[-4,19,4,19],[4,19,4,17],[4,17,5,16],[5,16,10,16],[10,16,11,17],[11,17,13,15],[13,15,14,16],[14,16,18,16]]
#geogebra 맵이라서 실제로는 이거의 200배 확대해서 씀
MAP_DUPLICATION = 200
LINE_THICKNESS = 10
MAX_TICK = 1800
MAX_VIEW_LOG = 6
original_map_path = [[0,0],[0,8],[7,8],[7,12],[-8,12],[-8,14],[-2,17],[-5,19.5],[4.5,19.5],[4.5,16.5],[10,16.5],[11,17.5],[13,15.5],[14,16.5],[18,16.15]]
map_lines, map_path = MAP_DUPLICATION*np.array(original_map_lines), MAP_DUPLICATION*np.array(original_map_path)


cam = [0,0,0]
focus_key = [False,False,False,False]
key_type = ['↑','←','→','↺']
key_index = [2,1,3,0]
focus_booster = 0
key_state = [False, False, False, False]#위, 왼쪽, 오른쪽, 드리프트(z)
game_tick = 0
generation = 0
spector = True#True일때 가장 빠른 차 따라감
max_time = 1800
max_distance = 0
last_max_point = 0

log = []

called = 0

class Car:
    def __init__(self, neural_network = None):
        self.x = 0
        self.y = 0
        self.speed_x = 0
        self.speed_y = 0
        self.speed = 10
        self.angle = np.pi/2
        self.speed_angle = 0
        self.booster = 0
        self.neural_network = []
        if neural_network == None:
            self.neural_network = [np.random.randn(i[0],i[1]) for i in NEURAL_NETWORK_SIZE]
        else:
            self.neural_network = neural_network
        self.point = 0
        self.time = 0
        self.ended = False
        self.cleared = False
        self.key_state = [False, False, False, False]

    def car_think(self, ray_state):
        state = ray_state + [self.speed/MAX_SPEED,max(0,100-self.booster)/100]
        output = state[:]
        for i in range(len(NEURAL_NETWORK_SIZE)):
            output = list((output+[1])@self.neural_network[i])
            output = [max(0,i) for i in output]#relu
        key = [1,0,0,0]
        index = output.index(max(output))
        if index == 4:
            key = [bool(i) for i in key]
            return key
        key[1] = index%2
        key[2] = (index+1)%2
        key[3] = index//2
        key[0] = 1-key[3]
        key = [bool(i) for i in key]
        return key

    def car_input(self,state):
        self.key_state = state[:]
        if state[3]:
            if state[1]: 
                self.speed_angle-=0.01
                if self.booster<100:
                    self.booster+=5
            if state[2]: 
                self.speed_angle+=0.01
                if self.booster<100:
                    self.booster+=5
        else:
            if state[0]: 
                self.speed+=(MAX_SPEED-self.speed)*0.1
                if self.booster>100:
                    self.speed+=(MAX_SPEED*2-self.speed)*0.05
            if state[1]: self.speed_angle-=0.003
            if state[2]: self.speed_angle+=0.003
        return 0

    def car_update(self):
        global max_time
        if (not self.cleared) and (not self.ended):self.time += 1
        self.speed_x = 0.7*self.speed_x + 0.3*self.speed*np.cos(self.angle)
        self.speed_y = 0.7*self.speed_y + 0.3*self.speed*np.sin(self.angle)
        self.x += self.speed_x
        self.y += self.speed_y
        if self.check_car_crashed():
            self.ended = True
            self.point = self.calc_point()
        if self.x>=3600:
            self.cleared = True
            self.point = 2-self.time/6000

        self.speed = max(0,self.speed-0.1)
        self.angle += self.speed_angle
        self.speed_angle *= 0.9

        if 100<self.booster<=110: self.booster += 200
        if 190<=self.booster<200: self.booster =0
        if self.booster>0: self.booster -= 1
        
        return 0
    
    def sensor(self):
        result = []
        for sensor_angle in range(-60,61,30):
            result.append(MAX_SENSOR_DISTANCE/messure_distance(self.x, self.y, self.angle + sensor_angle/180*np.pi))
        return result

    def calc_point(self):
        min_d = np.inf
        distance = 0
        for i in range(len(map_path)-1):
            vx,vy,ux,uy = map_path[i+1][0]-map_path[i][0],map_path[i+1][1]-map_path[i][1],self.x-map_path[i][0],self.y-map_path[i][1]
            d = abs((vx*uy - vy*ux)/((vx**2 + vy**2)**0.5+0.0001))
            t = (vx*ux + vy*uy)/(vx**2 + vy**2)
            if t<-0.2 or t>1.2:
                continue
            if min_d>d:
                min_d = d
                distance = i+t
        point = distance/(len(map_path)-1)
        return point

    def get_car_position(self):
        return [self.x,self.y,self.angle]
    
    def get_car_bossting(self):
        return self.booster>100
    
    def get_car_points(self):
        rotate = [[np.cos(self.angle),-np.sin(self.angle)],[np.sin(self.angle),np.cos(self.angle)]]
        corners = [[30, 30, -30, -30],[15, -15, -15, 15]]
        return np.transpose(np.array(rotate)@np.array(corners))+[[self.x,self.y]]*4

    def check_car_crashed(self):
        points = self.get_car_points()
        for i in range(3):
            if line_cross_check(points[i][0],points[i][1],points[i-1][0],points[i-1][1]):
                return True
        return False
    def print_car(self,screen):
        print_rotated_image(screen,car_image,cam[2]-self.angle,self.x,self.y)
    

cars = [Car() for i in range(POPULATION)]
mycar = Car()

def new_pos(pos):
    new_arr = np.cross([[np.cos(cam[2]),np.sin(cam[2])],[-np.sin(cam[2]),np.cos(cam[2])]],[pos[0]-cam[0], pos[1]-cam[1]])
    return [new_arr[0]+SCREEN_WIDTH/2, new_arr[1]+SCREEN_HEIGHT/2]

def focusing(car):
    global cam, focus_booster, focus_key
    cam = car.get_car_position()
    focus_booster = car.booster
    focus_key = car.key_state

def print_rotated_image(screen,image,angle,x,y):
    rotated_image = pygame.transform.rotate(image,angle*180/np.pi).convert_alpha()
    position = new_pos([x,y])
    screen.blit(rotated_image, [position[0] - rotated_image.get_width()/2, position[1] - rotated_image.get_height()/2])

def mix_unit(network1, network2):#퍼셉트론만 섞음
    new_network = []
    for i in range(len(network1)):
        layer1,layer2 = network1[i],network2[i]
        new_layer = [[0]*len(i) for i in layer1]
        for j in range(NEURAL_NETWORK_SIZE[i][0]):
            for k in range(NEURAL_NETWORK_SIZE[i][1]):
                if np.random.random()<0.6:
                    new_layer[j][k] = layer1[j][k]
                else:
                    new_layer[j][k] = layer2[j][k]
                if np.random.random()<0.1:
                    new_layer[j][k] *= 1+0.01*np.random.normal()
                if np.random.random()<MUTATION_RATE:
                    new_layer[j][k] = np.random.randn()
        new_network.append(np.array(new_layer))
    return new_network

def reset():
    global cars,mycar,max_time,max_distance,last_max_point,map_lines,map_path
    for car in cars:
        if (not car.ended) and (not car.cleared):
            car.point = car.calc_point()
        if car.cleared:
            max_time = min(max_time,(2-car.point)*6000)
    for i in range(1,POPULATION):
        if cars[i].point == cars[0].point:
            cars[i].point = 0
    cars.sort(key=lambda car: -car.point)

    max_distance = min(100,max(max_distance,cars[0].point*100))
    if last_max_point < cars[0].point:
        if max_time < MAX_TICK:
            log.append([generation,str(round(max_time/60,2)) + 's'])
        else:
            log.append([generation,str(round(max_distance,2)) + '%'])
        if len(log)>2 and log[-1][1] == log[-2][1]: del log[-1]
    last_max_point = cars[0].point
            
    probability = [max((POPULATION - x)**2,0) for x in range(POPULATION)]
    probability[0]*=POPULATION/6
    probability[1]*=POPULATION/12
    normalized_probability = [i/sum(probability) for i in probability]
    new_cars = []
    for i in range(POPULATION):
        selected = np.random.choice(POPULATION, 2, replace=False, p=normalized_probability)        
        new_cars.append(Car(mix_unit(cars[selected[0]].neural_network,cars[selected[1]].neural_network)))
    new_cars[0] = Car(cars[0].neural_network)
    new_cars[1] = Car(mix_unit(cars[0].neural_network,cars[0].neural_network))
    for i in range(POPULATION//7):
        new_cars[-1-i] = Car()
    cars = new_cars[:]
    mycar = Car()
    # if generation>50:
    #     for i in range(len(map_lines)):
    #         map_lines[i][0] *= -1
    #         map_lines[i][2] *= -1
        
    #     for i in range(len(map_path)):
    #         map_path[i][0] *= -1
    # print(map_lines)

@njit
def line_cross_check(x1,y1,x2,y2):
    
    for line in map_lines:
        x3,y3,x4,y4 = line[0],line[1],line[2],line[3]
        det = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if det == 0:
            continue
        point = [((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/det, ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/det]
        if (x1 - x2)**2 + (y1 - y2)**2 < (x1+x2 - 2*point[0])**2 + (y1+y2 - 2*point[1])**2:
            continue
        if (x3 - x4)**2 + (y3 - y4)**2 < (x3+x4 - 2*point[0])**2 + (y3+y4 - 2*point[1])**2:
            continue
        return True
    return False


@njit
def messure_distance(x,y,angle):
    min_distance = MAX_SENSOR_DISTANCE
    x1,y1,x2,y2 = x,y,x + MAX_SENSOR_DISTANCE*np.cos(angle),y + MAX_SENSOR_DISTANCE*np.sin(angle)
    for line in map_lines:
        x3,y3,x4,y4 = line[0],line[1],line[2],line[3]
        det = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if det == 0:
            continue
        point = [((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/det, ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/det]
        if (x1 - x2)**2 + (y1 - y2)**2 < (x1+x2 - 2*point[0])**2 + (y1+y2 - 2*point[1])**2:
            continue
        if (x3 - x4)**2 + (y3 - y4)**2 < (x3+x4 - 2*point[0])**2 + (y3+y4 - 2*point[1])**2:
            continue
        min_distance = min(min_distance, ((x-point[0])**2 + (y-point[1])**2)**0.5)
    return min_distance


def draw_window():
    screen.fill('white')
    
    #draw vertical lines
    for i in range(7):
        pygame.draw.line(screen, 'gray', new_pos([(cam[0]//100)*100 + i*100, cam[1]-SCREEN_HEIGHT]), new_pos([(cam[0]//100)*100 + i*100,cam[1]+SCREEN_HEIGHT]),1)
        pygame.draw.line(screen, 'gray', new_pos([(cam[0]//100)*100 - i*100, cam[1]-SCREEN_HEIGHT]), new_pos([(cam[0]//100)*100 - i*100,cam[1]+SCREEN_HEIGHT]),1)
    for i in range(7):
        pygame.draw.line(screen, 'gray', new_pos([cam[0]-SCREEN_HEIGHT, (cam[1]//100)*100 + i*100]), new_pos([cam[0]+SCREEN_HEIGHT, (cam[1]//100)*100 + i*100]),1)
        pygame.draw.line(screen, 'gray', new_pos([cam[0]-SCREEN_HEIGHT, (cam[1]//100)*100 - i*100]), new_pos([cam[0]+SCREEN_HEIGHT, (cam[1]//100)*100 - i*100]),1)
    
    for line in map_lines:
        pygame.draw.line(screen, 'black', new_pos(line[:2]), new_pos(line[2:]),LINE_THICKNESS)

    for car in cars:
        car.print_car(screen)
    mycar.print_car(screen)

    pygame.draw.line(screen, 'black', [SCREEN_WIDTH/2-105,SCREEN_HEIGHT-50],[SCREEN_WIDTH/2+105,SCREEN_HEIGHT-50],10)
    pygame.draw.line(screen, 'white', [SCREEN_WIDTH/2-103,SCREEN_HEIGHT-50],[SCREEN_WIDTH/2+103,SCREEN_HEIGHT-50],8)
    pygame.draw.line(screen, 'green', [SCREEN_WIDTH/2-100,SCREEN_HEIGHT-50],[SCREEN_WIDTH/2-100+min(focus_booster,100)*2,SCREEN_HEIGHT-50],5)
    gen = big_font.render('GENERATION:'+str(generation),True,'black')
    screen.blit(gen, (50,50))
    time = small_font.render(str(round(game_tick/60,2)) + 's',True,'black')
    screen.blit(time, (50,100))
    if max_time < 1800:
        m_record = small_font.render('best_record:'+str(round(max_time/60,2)) + 's',True,'black')
    else:
        m_record = small_font.render('best_record:'+str(round(max_distance,2)) + '%',True,'black')
    screen.blit(m_record, (50,150))
    if spector:
        for i in range(min(MAX_VIEW_LOG,len(log))):
            log_text = credit_font.render('new record : ' + str(log[-1-i][0]) + "'s generation : " + log[-1-i][1],True,(70,70,70))
            screen.blit(log_text, (900, 20*MAX_VIEW_LOG+30-(i + MAX_VIEW_LOG-min(MAX_VIEW_LOG,len(log)))*20))
    for i in range(len(focus_key)):
        if focus_key[i]:
            pygame.draw.rect(screen,(0,0,0,128),(50+key_index[i]*55,620,50,50),border_radius=5)
        else:
            pygame.draw.rect(screen,(0,0,0,128),(50+key_index[i]*55,620,50,50),1,border_radius=5)
        key = small_font.render(key_type[i],True,'white')
        key_rect = key.get_rect()
        key_rect.center = (75+key_index[i]*55,640)
        screen.blit(key,key_rect)

    credit = credit_font.render('by 20230379 신준현',True,'gray')
    screen.blit(credit, (1000,650))
    pygame.display.update()

def update_state():
    global game_tick, generation
    game_tick += 1
    if game_tick <= MAX_TICK:
        for car in cars:
            if not car.ended:
                car.car_input(car.car_think(car.sensor()))
                car.car_update()
        if cars[0].time+120 <= game_tick and (spector or (not spector and mycar.time+120 <= game_tick)):
            game_tick = MAX_TICK
    else:
        reset()
        game_tick = 0
        generation += 1
    
    if not mycar.ended:
        mycar.car_update()
    if not spector:
        if not mycar.ended:
            mycar.car_input(key_state)
        focusing(mycar)
    else:
        focusing(cars[0])

def file_write(file, text, end='\n'):
    file.write(str(text)+end)

def save():
    save_file = open(SAVE_FILE_PATH, 'w')
    file_write(save_file, generation)
    file_write(save_file, len(log))
    for i in log:
        file_write(save_file, i[0],end=' ')
        file_write(save_file, i[1])
    
    file_write(save_file, POPULATION)
    for car in cars:
        for layer in car.neural_network:
            for i in layer.flatten():
                file_write(save_file, i,end=' ')
            file_write(save_file,'')
    save_file.close()
    pass

def load_able():
    save_file = open(SAVE_FILE_PATH,'r')
    file_check = save_file.readline()
    save_file.close()
    if file_check: return True
    else: return False

def load():
    global generation, log, POPULATION, cars
    print('save file loading...')
    save_file = open(SAVE_FILE_PATH,'r')
    generation = int(save_file.readline())
    len_log = int(save_file.readline())
    for i in range(len_log):
        save_log = save_file.readline().split()
        log.append([int(save_log[0]), save_log[1]])
    POPULATION = int(save_file.readline())
    for i in range(POPULATION):
        neural_network = []
        for j in range(len(NEURAL_NETWORK_SIZE)):
            neural_network.append(np.array(list(map(float,save_file.readline().split())))
                                  .reshape(NEURAL_NETWORK_SIZE[j][0],NEURAL_NETWORK_SIZE[j][1]))
        cars[i] = Car(neural_network)
    save_file.close()
    print('save file loaded!')

def main():
    clock = pygame.time.Clock()
    running = True
    global game_tick, spector, MAX_VIEW_LOG

    if load_able():
        load_checker = tk.messagebox.askquestion('Load Check','Do you want to load the save file?')
        if load_checker == 'yes':
            load()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    key_state[0] = True
                if event.key == pygame.K_LEFT:
                    key_state[1] = True
                if event.key == pygame.K_RIGHT:
                    key_state[2] = True
                if event.key == pygame.K_z:
                    key_state[3] = True
                if event.key == pygame.K_a:
                    spector = not spector
                if event.key == pygame.K_l:
                    MAX_VIEW_LOG = 25
                if event.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    print('save!')
                    save()

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    key_state[0] = False
                if event.key == pygame.K_LEFT:
                    key_state[1] = False
                if event.key == pygame.K_RIGHT:
                    key_state[2] = False
                if event.key == pygame.K_z:
                    key_state[3] = False
                if event.key == pygame.K_l:
                    MAX_VIEW_LOG = 6
        update_state()
        draw_window()
        clock.tick(60)

if __name__ == '__main__':
    main()
pygame.quit()
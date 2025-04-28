import pygame
import sys 
import time
import run_ace
import ace_data_funcs
import numpy as np
import os
import shutil

brier_score = np.sum((np.asarray([0,0,1]) - np.asarray([0.3537731,  0.3958308,  0.25039607])) ** 2)/3
print(brier_score)
quit()


# initializing the constructor 
pygame.init() 
  
# screen resolution 
res = (1440,720) 
  
# opens up a window 
screen = pygame.display.set_mode(res) 
  
# white color 
color = (255,255,255) 
  
# light shade of the button 
color_light = (170,170,170) 
  
# dark shade of the button 
color_dark = (100,100,100) 
  
# stores the width of the 
# screen into a variable 
width = screen.get_width() 
  
# stores the height of the 
# screen into a variable 
height = screen.get_height() 
  
# defining a font 
smallfont = pygame.font.SysFont('arial',35) 
  
# rendering a text written in 
# this font 
quit_text = smallfont.render('quit' , True , color) 

continue_text = smallfont.render('continue' , True , color) 

run_num = 1

if os.path.exists('../../Data/ace_output/run' + str(run_num-1) + '_autoregressive_predictions_original.nc'):
    original_file  = '../../Data/ace_output/run' + str(run_num-1) + '_autoregressive_predictions_original.nc'
    copy_file = '../../Data/ace_output/run' + str(run_num-1) + '_autoregressive_predictions.nc'
  
    shutil.copyfile(original_file, copy_file)
    
if os.path.exists('../../Data/ace_output/run' + str(run_num-1) + '_restart_original.nc'):
    original_file  = '../../Data/ace_output/run' + str(run_num-1) + '_restart_original.nc'
    copy_file = '../../Data/ace_output/run' + str(run_num-1) + '_restart.nc'
  
    shutil.copyfile(original_file, copy_file)


west_temp = round(ace_data_funcs.get_loc_temp(47.61, 360 + (-122.33), run_num-1),2)

east_temp = round(ace_data_funcs.get_loc_temp(42.36, 360 + (-71.06), run_num-1),2)


# create rectangle 
input_rect = pygame.Rect(200, 200, 140, 32) 

user_text = ''

temp_change = 0

# input_box = pygame.Rect(100, 100, 140, 32)
# color_inactive = pygame.Color('lightskyblue3')
# color_active = pygame.Color('dodgerblue2')
# color = color_inactive
# active = False
# text = ''
# done = False

# if os.path.exists('/Users/nicojg/Data/ace_data/initialization/initialization_data.nc'):
#     print("File exists")
# else:
#     print("File does not exist")
# quit()

# quit_button_placement = [width-180,height-80,140,40]

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.image = pygame.image.load("../../Data/ace_output/figures/run" + str(run_num-1) + "_temps.png")
        self.image = pygame.transform.scale(self.image, (900, 450))
        self.rect = self.image.get_rect()
        self.rect.center = (520, 320)
 
    def update(self):
        pressed_keys = pygame.key.get_pressed()
       #if pressed_keys[K_UP]:
            #self.rect.move_ip(0, -5)
       #if pressed_keys[K_DOWN]:
            #self.rect.move_ip(0,5)
         
        # if self.rect.left > 0:
        #       if pressed_keys[K_LEFT]:
        #           self.rect.move_ip(-5, 0)
        # if self.rect.right < SCREEN_WIDTH:        
        #       if pressed_keys[K_RIGHT]:
        #           self.rect.move_ip(5, 0)
 
    def draw(self, surface):
        surface.blit(self.image, self.rect) 

    def update_image(self, image_path):
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (900, 450))

P1 = Player()

ace_data_funcs.update_plot(run_num-1)

P1.update_image("../../Data/ace_output/figures/run" + str(run_num-1) + "_temps.png")

class InputBox:
    def __init__(self, x, y, width, height, text=''):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color_light 
        self.text = text
        self.font = pygame.font.Font(None, 32)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the color of the input box when selected
            self.color = color_light  if self.active else color_dark
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    # onenter(self.text)
                    global temp_change
                    temp_change = self.text
                    # ace_data_funcs.modify_ace_input(47.61, 360 + (-122.33), temp_change,run_num-1)
                    ace_data_funcs.modify_ace_input(13.231925, 360 + (-141.137174), temp_change,run_num-1)
                    ace_data_funcs.update_plot(run_num-1)
                    P1.update_image("../../Data/ace_output/figures/run" + str(run_num-1) + "_temps.png")
                    self.text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode

    def draw(self, screen):
        # Render the text.
        text_surface = self.font.render(self.text, True, self.color)
        # Resize the box if the text is too long.
        width = max(200, text_surface.get_width()+10)
        self.rect.w = width
        # Blit the text.
        screen.blit(text_surface, (self.rect.x+5, self.rect.y+5))
        # Draw the rectangle.
        pygame.draw.rect(screen, self.color, self.rect, 2)

input_box = InputBox(1125, 200, 140, 32) # position (x and y) and size (width and height)


while True: 
      
    for ev in pygame.event.get(): 

        input_box.handle_event(ev)
          
        if ev.type == pygame.QUIT: 
            pygame.quit()
            sys.exit()
              
        #checks if a mouse is clicked 
        if ev.type == pygame.MOUSEBUTTONDOWN: 
              
            #if the mouse is clicked on the 
            # button the game is terminated 
            if width-180 <= mouse[0] <= width-180+140 and height-80 <= mouse[1] <= height-80+40: 
                pygame.quit()
                sys.exit()
            # Check if the mouse is hovered on a button
            if -1*width+1480 <= mouse[0] <= -1*width+1480+140 and height-80 <= mouse[1] <= height-80+40: 
                # Render and display the "Running ACE" text
                running_text = smallfont.render('Running ACE...' , True , color)
                screen.blit(running_text, (width-1350+20, height-700+10))
                pygame.display.flip()
                
                # Run the ACE program
                ace_data_funcs.create_yaml(run_num)

                run_ace.inf_ace(run_num)
                run_num += 1

                west_temp = round(ace_data_funcs.get_loc_temp(47.61, 360 + (-122.33), run_num-1),2)

                east_temp = round(ace_data_funcs.get_loc_temp(42.36, 360 + (-71.06), run_num-1),2)

                ace_data_funcs.update_plot(run_num-1)

                P1.update_image("../../Data/ace_output/figures/run" + str(run_num-1) + "_temps.png")

                # Clear the button area
                pygame.draw.rect(screen, (0, 70, 0), [width-1350, height-700, 140, 40])
                pygame.display.flip()
            # If the user clicked on the input_box rect.
        #     if input_box.collidepoint(ev.pos):
        #         # Toggle the active variable.
        #         active = not active
        #     else:
        #         active = False
        
        # if ev.type == pygame.KEYDOWN:
        #     if active:
        #         if ev.key == pygame.K_RETURN:
        #             print(text)
        #             text = ''
        #         elif ev.key == pygame.K_BACKSPACE:
        #             text = text[:-1]
        #         else:
        #             text += ev.unicode
            if ev.type == pygame.KEYDOWN: 

                # Check for backspace 
                if ev.key == pygame.K_BACKSPACE: 

                    # get text input from 0 to -1 i.e. end. 
                    user_text = user_text[:-1] 

                # Unicode standard is used for string 
                # formation 
                else: 
                    user_text += ev.unicode
    # fills the screen with a color 
    screen.fill((0,70,0)) 



    P1.draw(screen) 

    input_box.draw(screen)

    # stores the (x,y) coordinates into 
    # the variable as a tuple 
    mouse = pygame.mouse.get_pos() 
      
    # if mouse is hovered on a button it 
    # changes to lighter shade  
    if width-180 <= mouse[0] <= width-180+140 and height-80 <= mouse[1] <= height-80+40: 
        pygame.draw.rect(screen,color_light,[width-180,height-80,140,40]) 
          
    else: 
        pygame.draw.rect(screen,color_dark,[width-180,height-80,140,40]) 
      
    # superimposing the text onto our button 
    screen.blit(quit_text,(width-180+40,height-80)) 

    mouse = pygame.mouse.get_pos() 
      
    # if mouse is hovered on a button it 
    # changes to lighter shade  
    if -1*width+1480 <= mouse[0] <= -1*width+1480+140 and height-80 <= mouse[1] <= height-80+40: 
        pygame.draw.rect(screen,color_light,[-1*width+1480,height-80,140,40]) 
          
    else: 
        pygame.draw.rect(screen,color_dark,[-1*width+1480,height-80,140,40]) 
      
    # superimposing the text onto our button 
    screen.blit(continue_text,(-1*width+1480+5,height-80)) 

    west_temperature_text = smallfont.render(f'Seattle Temp: {west_temp}°K', True, color)
    smaller_west_temperature_text = pygame.transform.scale(west_temperature_text, (int(west_temperature_text.get_width() * .75), int(west_temperature_text.get_height() * .75)))
    screen.blit(smaller_west_temperature_text, (width - 400, 10))

    east_temperature_text = smallfont.render(f'Boston Temp: {east_temp}°K', True, color)
    smaller_east_temperature_text = pygame.transform.scale(east_temperature_text, (int(east_temperature_text.get_width() * .75), int(east_temperature_text.get_height() * .75)))
    screen.blit(smaller_east_temperature_text, (width - 400, 40))

    score_text = smallfont.render(f'Current Score: {np.abs(np.round(east_temp-west_temp, 2))}°K', True, color)
    score_text = pygame.transform.scale(score_text, (int(score_text.get_width() * .75), int(score_text.get_height() * .75)))
    screen.blit(score_text, (width - 400, 70))

    running_text = smallfont.render('Enter Temp change:' , True , color)
    screen.blit(running_text, (width-395, height-600+10))

    temp_text = smallfont.render('Temp Modification: ' + str(temp_change) , True , color)
    screen.blit(temp_text, (width-395, height-200+10))

    pygame.display.flip()

    # pygame.draw.rect(screen, color, input_rect) 
  
    # text_surface = smallfont.render(user_text, True, (255, 255, 255)) 

    #  # render at position stated in arguments 
    # screen.blit(text_surface, (1000, 60)) 
      
    # # set width of textfield so that text cannot get 
    # # outside of user's text input 
    # input_rect.w = max(100, text_surface.get_width()+10) 
    # pygame.display.flip()

    # txt_surface = smallfont.render(text, True, color)
    # # Resize the box if the text is too long.
    # width = max(200, txt_surface.get_width()+10)
    # input_box.w = width
    # # Blit the text.
    # screen.blit(txt_surface, (input_box.x+5, input_box.y+5))
    # # Blit the input_box rect.
    # pygame.draw.rect(screen, color, input_box, 2)

    # updates the frames of the game 
    pygame.display.update() 
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

## F'n to add players' names and numbers to tracking data
def add_names(tracking):
    GAME_ID=tracking.at[0,'game_id']
    home=pd.read_json(f'data/00{GAME_ID}.json')['events'][0]['home']
    visitor=pd.read_json(f'data/00{GAME_ID}.json')['events'][0]['visitor']

    # creates the players list with the home players
    players = home["players"]
    # Then add on the visiting players
    players.extend(visitor["players"])
    len(players)

    # initialize new dictionary
    id_dict = {}

    # Add the values we want
    for player in players:
        id_dict[player['playerid']] = [player["firstname"]+" "+player["lastname"],
                                   player["jersey"]]

    id_dict.update({-1: ['ball', np.nan]})

    tracking["player_name"] = tracking.player_id.map(lambda x: id_dict[x][0])
    tracking["player_jersey"] = tracking.player_id.map(lambda x: id_dict[x][1])
    

    
    
## Revised f'ns for fixing shot data  
def fix_shot_times_edited (shots,tracking):
    for i in range(0,len(shots)):
        
        a = tracking.loc[tracking['quarter']==shots.at[i,'QUARTER']]
        times = a[['game_clock']]
        time_to_check=shots.at[i,'EVENTTIME']##The time on the scoreboard when the shot is taken
        d=times.assign(diff= lambda x: abs(x.game_clock-time_to_check))##Find the absolute diff between
        ##Assign the game_clock value w the minimun diff to 'SHOT_TIME'
        low=d.loc[d['diff']==min(d['diff'])].reset_index(drop=True)
        shots.at[i,'SHOT_TIME']=low.at[0,'game_clock']

        
        
def fix_shot_locations (shots,tracking):
    for i in range(0,len(shots)):
        
        a = tracking.loc[tracking['game_clock']==shots.at[i,'SHOT_TIME']]
        ball_x=a.loc[a['player_id']==-1].reset_index(drop=True).at[0,'x_loc']
        ball_y=a.loc[a['player_id']==-1].reset_index(drop=True).at[0,'y_loc']
        shots.at[i,'LOC_X']=ball_x
        shots.at[i,'LOC_Y']=ball_y
        
        
        

# Function to find the distance between players
def player_dist(player_a, player_b):

    result = []
    for i in range(len(player_a)):
        try:
            dist = euclidean(player_a.iloc[i], player_b.iloc[i])
        except:
            dist = 0
        result.append(dist)
        
    return result


## F'n to draw court
def draw_half_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


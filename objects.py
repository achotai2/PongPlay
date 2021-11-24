import numpy
import atexit
import datetime

from agent_objects import AgentOrange

# Dimensions of screen.
screenWidth  = 800
screenHeight = 600

# Set up Turtle screen stuff----------------------------------------------------
import turtle

wn = turtle.Screen()
wn.title( "Squares and Stuff" )
wn.bgcolor( "black" )
wn.setup( width = screenWidth, height = screenHeight )
wn.tracer( 0 )

boxColour  = 1
objWidth   = 100
objHeight  = 100
objCenterX = 0
objCenterY = 0
box = turtle.Turtle()
box.speed( 0 )
box.shape( "square" )
box.color( "red" )
box.shapesize( stretch_wid = objWidth / 10, stretch_len = objHeight / 10 )
box.penup()
box.goto( objCenterX, objCenterY )

senseResX = 10
senseResY = 10
sensePosX = 100
sensePosY = 100
senseOrgan = turtle.Turtle( )
senseOrgan.speed( 0 )
senseOrgan.shape( "square" )
senseOrgan.color( "blue" )
senseOrgan.shapesize( stretch_wid = senseResX / 10, stretch_len = senseResY / 10 )
senseOrgan.penup()
senseOrgan.goto( sensePosX, sensePosY )

# Create agent------------------------------------------------------------------
Agent1 = AgentOrange( 'Agent', senseResX, senseResY )

# Prepare log file--------------------------------------------------------------
timeStep = 0
current_date_and_time_string = str( datetime.datetime.now() )
file_name =  "Logs/" + current_date_and_time_string + ".txt"
file = open(file_name, 'a')
file.write( "Program Start Time: " + str( datetime.datetime.now() ) )
file.write( "\n" )

def exit_handler():
# Upon program exit appends end time and closes log file.

    log_data = []
    log_data.append( "-------------------------------------------------------" )

    Agent1.ReturnEndState( log_data )

    log_data.append( "\n" + "Program End Time: " + str( datetime.datetime.now() ) )

    for line in log_data:
        file.write( line )
        file.write( "\n" )

    file.close()

atexit.register(exit_handler)

# Main program loop----------------------------------------------------------------
while True:

    wn.update()         # Screen update

#    if boxColour == 1:
#        boxColour = 2
#        box.color( "green" )
#    elif boxColour == 2:
#        boxColour = 3
#        box.color( "blue" )
#    elif boxColour == 3:
#        boxColour = 1
#        box.color( "red" )

    # Create text file to store log data.
    timeStep += 1
    log_data = []
    log_data.append( "-------------------------------------------------------" )
    log_data.append( "Time Step: " + str( timeStep ) )

    # Run agent brain and get motor vector.
    organVector = Agent1.Brain( objCenterX, objCenterY, objWidth, objHeight, boxColour, sensePosX, sensePosY, log_data )

    # Write log data to text file.
    for line in log_data:
        file.write( line )
        file.write( "\n" )

    # Move agents sense organ accordingt to returned vector.
    sensePosX += organVector[ 0 ]
    sensePosY += organVector[ 1 ]
    senseOrgan.goto( sensePosX, sensePosY )

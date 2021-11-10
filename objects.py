import numpy

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

while True:
# Main game loop----------------------------------------------------------------

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

    organVector = Agent1.Brain( objCenterX, objCenterY, objWidth, objHeight, boxColour, sensePosX, sensePosY )

    sensePosX += organVector[ 0 ]
    sensePosY += organVector[ 1 ]
    senseOrgan.goto( sensePosX, sensePosY )

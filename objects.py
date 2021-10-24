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
objWidth   = 10
objHeight  = 10
objCenterX = 100
objCenterY = 100
box = turtle.Turtle()
box.speed( 0 )
box.shape( "square" )
box.color( "red" )
box.shapesize( stretch_wid = objWidth, stretch_len = objHeight )
box.penup()
box.goto( objCenterX, objCenterY )

# Create agent------------------------------------------------------------------
Agent1 = AgentOrange( 'Agent' )

while True:
# Main game loop----------------------------------------------------------------

    wn.update()         # Screen update

    if boxColour == 1:
        boxColour = 2
        box.color( "green" )
    elif boxColour == 2:
        boxColour = 3
        box.color( "blue" )
    elif boxColour == 3:
        boxColour = 1
        box.color( "red" )
    Agent1.Brain( objCenterX, objCenterY, objWidth, objHeight, boxColour )

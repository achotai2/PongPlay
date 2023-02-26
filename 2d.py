import turtle
import atexit
from agent_run import AgentRun
from useful_functions import Within
from logs_yo import Logging

#-------------------------------------------------------------------------------

screenHeight = 600          # Used in setting up screen and encoders
screenWidth = 800

# Set up turtle screen.
wn = turtle.Screen( )
wn.title( "2D" )
wn.bgcolor( "black" )
wn.setup( width = screenWidth, height = screenHeight )
wn.tracer( 0 )

maxClockTime  = 2
startTheClock = maxClockTime

# Set up sense organ.
senseResX = 20
senseResY = 20
sensePosX = 0
sensePosY = 0
senseOrgan = turtle.Turtle( )
senseOrgan.speed( 0 )
senseOrgan.shape( "square" )
senseOrgan.color( "white" )
senseOrgan.shapesize( stretch_wid = senseResX / 10, stretch_len = senseResY / 10 )
senseOrgan.penup()
senseOrgan.goto( sensePosX, sensePosY )

# Set up objects......
objWidth   = 10
objHeight  = 10
objectList = []

# Set up agent.
objColour  = 2
objCenterX = 0
objCenterY = 0
objectList.append( [ objCenterX, objCenterY, objWidth, objHeight, objColour ] )
agentDraw = turtle.Turtle( )
agentDraw.speed( 0 )
agentDraw.shape( "square" )
agentDraw.color( "blue" )
agentDraw.shapesize( stretch_wid = objWidth / 10, stretch_len = objHeight / 10 )
agentDraw.penup()
agentDraw.goto( objCenterX, objCenterY )

# Set up den.
#objColour  = 1
#objCenterX = -100
#objCenterY = -100
#objectList.append( [ objCenterX, objCenterY, objWidth, objHeight, objColour ] )
#denDraw = turtle.Turtle()
#denDraw.speed( 0 )
#denDraw.shape( "square" )
#denDraw.color( "green" )
#denDraw.shapesize( stretch_wid = objWidth / 10, stretch_len = objHeight / 10 )
#denDraw.penup()
#denDraw.goto( objCenterX, objCenterY )

# Set up Enemy.
objColour  = 0
objCenterX = 100
objCenterY = 100
objectList.append( [ objCenterX, objCenterY, objWidth, objHeight, objColour ] )
enemyDraw = turtle.Turtle()
enemyDraw.speed( 0 )
enemyDraw.shape( "square" )
enemyDraw.color( "red" )
enemyDraw.shapesize( stretch_wid = objWidth / 10, stretch_len = objHeight / 10 )
enemyDraw.penup()
enemyDraw.goto( objCenterX, objCenterY )

#-------------------------------------------------------------------------------

# Functions
def UpdateObjectDraw():
# Use the updated positions of the objects to update their drawn positions.

    agentDraw.setx( objectList[ 0 ][ 0 ] )
    agentDraw.sety( objectList[ 0 ][ 1 ] )

    enemyDraw.setx( objectList[ 1 ][ 0 ] )
    enemyDraw.sety( objectList[ 1 ][ 1 ] )

def MoveEnemy():
# Move the enemy towards the player.

    if objectList[ 0 ][ 0 ] > objectList[ 1 ][ 0 ]:
        objectList[ 1 ][ 0 ] += 10
    elif objectList[ 0 ][ 0 ] < objectList[ 1 ][ 0 ]:
        objectList[ 1 ][ 0 ] -= 10

    if objectList[ 0 ][ 1 ] > objectList[ 1 ][ 1 ]:
        objectList[ 1 ][ 1 ] += 10
    elif objectList[ 0 ][ 1 ] < objectList[ 1 ][ 1 ]:
        objectList[ 1 ][ 1 ] -= 10

def CheckCollision( theThing, maxThing ):
# If enemy and player are at same position then refresh positions and send a negative signal to agent.
# Upon a collision we feed the agent random input for maxClockTime time steps.

    if theThing < maxThing:
        theThing += 1

    if Within( objectList[ 0 ][ 0 ], objectList[ 1 ][ 0 ] - objectList[ 1 ][ 2 ], objectList[ 1 ][ 0 ] + objectList[ 1 ][ 2 ], True ):
        if Within( objectList[ 0 ][ 1 ], objectList[ 1 ][ 1 ] - objectList[ 1 ][ 3 ], objectList[ 1 ][ 1 ] + objectList[ 1 ][ 3 ], True ):
            ResetObjects()
            theThing = 0

    return theThing

def ResetObjects():
# Reset the objects back to start position.

    objectList[ 0 ][ 0 ] = 0
    objectList[ 0 ][ 1 ] = 0
    objectList[ 1 ][ 0 ] = 100
    objectList[ 1 ][ 1 ] = 100

def agent_up( ):
    y = agentDraw.ycor()
    if y < 290 - 20:
        y += 20
        agentDraw.sety( y )

def agent_down( ):
    y = agentDraw.ycor()
    if y > -290 + 20:
        y -= 20
        agentDraw.sety( y )

def agent_right( ):
    x = agentDraw.xcor()
    if x < 390 - 20:
        x += 20
        agentDraw.setx( x )

def agent_left( ):
    x = agentDraw.xcor()
    if x > -390 + 20:
        x -= 20
        agentDraw.setx( x )

def exit_handler():
# Upon program exit collects data for Cell-Report log file, and produces the final plot.

    logFile.WhenExit( [ agentRun ] )

#-------------------------------------------------------------------------------

# Create agent.
agentRun = AgentRun( "Run", senseResX, senseResY, 2, 0 )

# Prepare log and report files.
logFile = Logging( [ agentRun.ID ] )

atexit.register( exit_handler )

#-------------------------------------------------------------------------------

while True:
# Main game loop

    startTheClock = CheckCollision( startTheClock, maxClockTime )
    if startTheClock < maxClockTime:
        randomInput = True
    else:
        randomInput = False

    UpdateObjectDraw()

    wn.update()         # Screen update

    logFile.AddToTimeStep()

    # Run agent brain and get motor vector.
    organVector = agentRun.Brain( objectList, sensePosX, sensePosY, 0.0, randomInput )

    # Accumulate the active cells and segments and input into report data.
    logFile.AccumulateReportData( [ agentRun ], [ sensePosX, sensePosY ] )

    # Update the sense organ position.
    sensePosX += organVector[ 0 ]
    sensePosY += organVector[ 1 ]
    senseOrgan.goto( sensePosX, sensePosY )

    # Move the enemy.
#    if not randomInput:
#        MoveEnemy()

    # Write segment data to individual files for report.
    logFile.WriteDataToFiles( [ agentRun ] )

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
sensePos = []
sensePos.append( 0 )
sensePos.append( 0 )
senseOrgan = turtle.Turtle( )
senseOrgan.speed( 0 )
senseOrgan.shape( "square" )
senseOrgan.color( "white" )
senseOrgan.shapesize( stretch_wid = senseResX / 10, stretch_len = senseResY / 10 )
senseOrgan.penup()
senseOrgan.goto( sensePos[ 0 ], sensePos[ 1 ] )

# Set up objects......
objWidth   = 10
objHeight  = 10
objectList = []

# Set up agent.
objColour  = 2
objCenterX = 0
objCenterY = 0
agentVector = [ 0, 0, 0, 0 ]
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

    senseOrgan.goto( sensePos[ 0 ], sensePos[ 1 ] )

def MoveEnemy():
# Move the enemy towards the player.

    enemyMoveAmount = 10

    if objectList[ 0 ][ 0 ] > objectList[ 1 ][ 0 ]:
        objectList[ 1 ][ 0 ] += enemyMoveAmount
    elif objectList[ 0 ][ 0 ] < objectList[ 1 ][ 0 ]:
        objectList[ 1 ][ 0 ] -= enemyMoveAmount

    if objectList[ 0 ][ 1 ] > objectList[ 1 ][ 1 ]:
        objectList[ 1 ][ 1 ] += enemyMoveAmount
    elif objectList[ 0 ][ 1 ] < objectList[ 1 ][ 1 ]:
        objectList[ 1 ][ 1 ] -= enemyMoveAmount

def MoveAgent( desiredVector ):
# Move the agent and its sense organ based on desiredVector.

    agentMoveAmount = 20

    prePosition = []
    prePosition.append( objectList[ 0 ][ 0 ] )
    prePosition.append( objectList[ 0 ][ 1 ] )

    # Update the agents position.
    if desiredVector[ 2 ] > 0:
        objectList[ 0 ][ 0 ] += agentMoveAmount
        if objectList[ 0 ][ 0 ] > screenWidth / 2:
            objectList[ 0 ][ 0 ] = screenWidth / 2
    elif desiredVector[ 2 ] < 0:
        objectList[ 0 ][ 0 ] -= agentMoveAmount
        if objectList[ 0 ][ 0 ] < -screenWidth / 2:
            objectList[ 0 ][ 0 ] = -screenWidth / 2

    if desiredVector[ 3 ] > 0:
        objectList[ 0 ][ 1 ] += agentMoveAmount
        if objectList[ 0 ][ 1 ] > screenHeight / 2:
            objectList[ 0 ][ 1 ] = screenHeight / 2
    elif desiredVector[ 3 ] < 0:
        objectList[ 0 ][ 1 ] -= agentMoveAmount
        if objectList[ 0 ][ 1 ] < -screenHeight / 2:
            objectList[ 0 ][ 1 ] = -screenHeight / 2

    # Update the sense organ positions.
    sensePos[ 0 ] += desiredVector[ 0 ]
    sensePos[ 1 ] += desiredVector[ 1 ]

    # Calculate the agents vector.
    newVector = []
    newVector.append( desiredVector[ 0 ] )
    newVector.append( desiredVector[ 1 ] )
    newVector.append( objectList[ 0 ][ 0 ] - prePosition[ 0 ] )
    newVector.append( objectList[ 0 ][ 1 ] - prePosition[ 1 ] )

    return newVector

def CheckCollision( theThing, maxThing ):
# If enemy and player are at same position then refresh positions and send a negative signal to agent.
# Upon a collision between agent and enemy we feed the agent random input for maxClockTime time steps.

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

def exit_handler():
# Upon program exit collects data for Cell-Report log file, and produces the final plot.

    logFile.WhenExit( [ agentRun ] )

#-------------------------------------------------------------------------------

# Create agent.
agentRun = AgentRun( "Run", senseResX, senseResY, screenWidth, screenHeight )

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

    wn.update()         # Screen update

#    logFile.AddToTimeStep()

    # Run agent brain and get motor vector.
    desiredVector = agentRun.Brain( objectList, sensePos[ 0 ], sensePos[ 1 ], agentVector, [ objectList[ 0 ][ 0 ], objectList[ 0 ][ 1 ] ], 0.0, randomInput )

    # Accumulate the active cells and segments and input into report data.
#    logFile.AccumulateReportData( [ agentRun ], sensePos )

    # Move the Agent.
    agentVector = MoveAgent( desiredVector )

    # Move the enemy.
    if not randomInput:
        MoveEnemy()

    UpdateObjectDraw()

    # Write segment data to individual files for report.
#    logFile.WriteDataToFiles( [ agentRun ] )

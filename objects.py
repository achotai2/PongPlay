import numpy
import atexit
import datetime
import os
import matplotlib.pyplot as plt
from bisect import bisect_left
from logs_yo import Logging
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

inputNoisePct = 0.0

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
Agent1 = AgentOrange( "Objects", senseResX, senseResY, 2, 1 )

# Prepare log and report files--------------------------------------------------
logFile = Logging( [ Agent1.ID ] )

def exit_handler():
# Upon program exit collects data for Cell-Report log file, and produces the final plot.

    logFile.WhenExit( [ Agent1 ] )

atexit.register( exit_handler )

# Main program loop-------------------------------------------------------------
while True:

    wn.update()         # Screen update

    logFile.AddToTimeStep()

    if int( logFile.timeStep / 1000 ) % 3 == 1:
        boxColour = 2
        box.color( "green" )
    elif int( logFile.timeStep / 1000 ) % 3 == 2:
        boxColour = 3
        box.color( "blue" )
    elif int( logFile.timeStep / 1000 ) % 3 == 0:
        boxColour = 1
        box.color( "red" )

    # Run agent brain and get motor vector.
    organVector = Agent1.Brain( objCenterX, objCenterY, objWidth, objHeight, boxColour, sensePosX, sensePosY, inputNoisePct )

    # Accumulate the active cells and segments and input into report data.
    logFile.AccumulateReportData( [ Agent1 ], [ sensePosX, sensePosY, boxColour ] )

    # Move agents sense organ accordingt to returned vector.
    sensePosX += organVector[ 0 ]
    sensePosY += organVector[ 1 ]
    senseOrgan.goto( sensePosX, sensePosY )

    # Write segment data to individual files for report.
    logFile.WriteDataToFiles( [ Agent1 ] )

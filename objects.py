import numpy
import atexit
import datetime
import os
import matplotlib.pyplot as plt

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

# Prepare log and report files--------------------------------------------------
timeStep = 0
start_date_and_time_string = str( datetime.datetime.now() )
os.mkdir( "./Logs/" + start_date_and_time_string )

log_file_name    =  "Logs/" + start_date_and_time_string + "/Log" + ".txt"
#cell_report_file_name =  "Logs/" + start_date_and_time_string + "-Cell-Report" + ".txt"

log_file = open( log_file_name, 'a' )
log_file.write( "Program Start Time: " + str( start_date_and_time_string ) )
log_file.write( "\n" )
log_file.close()

position_data        = [ 0, [], [] ] * 4       # top left, top right, bottom left, bottom right. [ total count, [ cells ], [ count ] ]
vector_data          = [ 0, [], [] ] * 16      # All the vectors. [ total count, [ segments ], [ count ] ]
graphY1NumActiveCells = []
graphY2NumActiveSegs  = []
graphY3NumValidSegs   = []
graphXTimeSteps       = []

def WriteDataToFiles( timeStep ):
# Go through all returned segment data and write it to separate segment files.

    # Create text file to store log data.
    log_data = []
    log_file = open( log_file_name, 'a' )
    log_data.append( "-------------------------------------------------------" )
    log_data.append( "Time Step: " + str( timeStep ) )

    new_data = Agent1.GetLogData()
    for entry in new_data:
        log_data.append( entry )

    # Write log data to text file.
    for line in log_data:
        log_file.write( line )
        log_file.write( "\n" )
    log_file.close()

    # Write segment data to individual reports.
# CAREFUL WITH COMMENTING OUT THIS FUNCTION AS EACH SEGMENTS LOCAL DATA STORAGE ONLY REFRESHED HERE.
    segment_report_data = Agent1.ReturnSegmentState( timeStep )

    for index, segment_data in enumerate( segment_report_data ):

        segment_report_file_name =  "Logs/" + start_date_and_time_string + "/Segment_" + str( index ) + ".txt"
        segment_report_file = open( segment_report_file_name, 'a' )

        for line in segment_data:
            segment_report_file.write( line )
            segment_report_file.write( "\n" )
        segment_report_file.close()

def AccumulateReportData( timeStep, organVector ):
# Accumulate the active cells and segments and input into report data.

    activeCells, activeSegments, numValidSegments  = Agent1.GetReportData()

    if sensePosX == 100 and sensePosY == 100:
        posIndex = 0
    elif sensePosX == 100 and sensePosY == -100:
        posIndex = 1
    elif sensePosX == -100 and sensePosY == 100:
        posIndex = 2
    elif sensePosX == -100 and sensePosY == -100:
        posIndex = 3

    # Data for graph.
    # x-axis and y-axis values
    graphY1NumActiveCells.append( len( activeCells ) )
    graphY2NumActiveSegs.append( len( activeSegments ) )
    graphY3NumValidSegs.append( numValidSegments )
    graphXTimeSteps.append( timeStep )

def exit_handler():
# Upon program exit appends end time and closes log file.

    log_file = open( log_file_name, 'a' )
    log_file.write( "\n" )
    log_file.write( "-------------------------------------------------------" )
    log_file.write( "\n" + "Program End Time: " + str( datetime.datetime.now() ) )
    log_file.close()

#    cell_report_data = []
#    cell_report_data.append( start_date_and_time_string )
#    cell_report_data.append( "-------------------------------------------------------" )

#    Agent1.ReturnEndState( cell_report_data )

#    cell_report_data.append( "\n" + "Program End Time: " + str( datetime.datetime.now() ) )

#    cell_report_file = open( cell_report_file_name, 'w' )
#    for line in cell_report_data:
#        cell_report_file.write( line )
#        cell_report_file.write( "\n" )
#    cell_report_file.close()

    # --------- Plot the graph. ----------------
    # plotting the points
    plt.plot( graphXTimeSteps, graphY1NumActiveCells, label = "# Active F-Cells" )
    plt.plot( graphXTimeSteps, graphY2NumActiveSegs, label = "# Active Segments" )
    plt.plot( graphXTimeSteps, graphY3NumValidSegs, label = "# Valid Segments" )
    # naming the x axis
    plt.xlabel('Time Steps')
    # naming the y axis
    plt.ylabel('# of Active')
    # giving a title to my graph
    plt.title('# F-Cells and Segments Over Time')
    plt.legend()
    # function to show the plot
    plt.savefig( "Logs/" + start_date_and_time_string + "/Plot_Data.png" )
    plt.show()

atexit.register( exit_handler )

# Main program loop-------------------------------------------------------------
while True:

    wn.update()         # Screen update

    timeStep += 1

    if int( timeStep / 300 ) % 3 == 1:
        boxColour = 2
        box.color( "green" )
    elif int( timeStep / 300 ) % 3 == 2:
        boxColour = 3
        box.color( "blue" )
    elif int( timeStep / 300 ) % 3 == 0:
        boxColour = 1
        box.color( "red" )

    # Run agent brain and get motor vector.
    organVector = Agent1.Brain( objCenterX, objCenterY, objWidth, objHeight, boxColour, sensePosX, sensePosY )

    AccumulateReportData( timeStep, organVector )

    # Move agents sense organ accordingt to returned vector.
    sensePosX += organVector[ 0 ]
    sensePosY += organVector[ 1 ]
    senseOrgan.goto( sensePosX, sensePosY )

    # Write segment data to individual files for report.
    WriteDataToFiles( timeStep )

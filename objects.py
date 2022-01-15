import numpy
import atexit
import datetime
import os
import matplotlib.pyplot as plt
from bisect import bisect_left

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
log_file = open( log_file_name, 'a' )
log_file.write( "Program Start Time: " + str( start_date_and_time_string ) )
log_file.write( "\n" )
log_file.close()

cellData              = []
stateData             = []
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

def AccumulateReportData( timeStep, boxColour ):
# Accumulate the active cells and segments and input into report data.

    activeCellsBool, activeSegments, numValidSegments = Agent1.GetReportData()

    # A list of indices of all active cells.
    activeCells = []
    for index, cell in enumerate( activeCellsBool ):
        if cell:
            activeCells.append( index )

    # Gather state data.
    stateIndex = 0
    currentState  = 0
    while stateIndex < len( stateData ):
        if stateData[ stateIndex ][ 1 ] == sensePosX and stateData[ stateIndex ][ 2 ] == sensePosY and stateData[ stateIndex ][ 3 ] == boxColour:
            stateData[ stateIndex ][ 0 ] += 1
            break
        stateIndex += 1
    if stateIndex == len( stateData ):
        stateData.append( [ 1, sensePosX, sensePosY, boxColour ] )
    currentState = stateIndex

    # Find active cell identification data.
    for i, aCell in enumerate( activeCellsBool ):
        if i < len( cellData ):
            if aCell:
                cellData[ i ][ 1 ] += 1
                entryIndex = 2
                while entryIndex < len( cellData[ i ] ):
                    if cellData[ i ][ entryIndex ][ 0 ] == currentState:
                        cellData[ i ][ entryIndex ][ 1 ] += 1
                        break
                    entryIndex += 1
                if entryIndex == len( cellData[ i ] ):
                    cellData[ i ].append( [ currentState, 1 ] )
        else:
            cellData.append( [ i, 0 ] )
            if aCell:
                cellData[ -1 ][ 1 ] += 1
                cellData[ -1 ].append( [ currentState, 1 ] )

    # Data for graph.
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

    cell_report_data = []
    cell_report_data.append( start_date_and_time_string )
    cell_report_data.append( "-------------------------------------------------------" )
    cell_report_data.append( "- \t Cell ID \t # Times Active \t ( State active in, % times active ) -" )
    cellData.sort( key = lambda cellData: cellData[ 1 ], reverse = True )
    for cell in cellData:
        state_data_str = ""
        entryIndex = 2
        while entryIndex < len( cell ):
            state_data_str += "( " + str( [ stateData[ cell[ entryIndex ][ 0 ] ][ i ] for i in range( 1, len( stateData[ cell[ entryIndex ][ 0 ] ] ) ) ] ) + ", " + str( int( cell[ entryIndex ][ 1 ] / stateData[ cell[ entryIndex ][ 0 ] ][ 0 ] * 100 ) ) + "% ), "
            entryIndex += 1
        cell_report_data.append( "\t" + str( cell[ 0 ] ) + "\t\t\t" + str( cell[ 1 ] ) + "\t\t" + state_data_str )
    cell_report_data.append( "\n" + "Program End Time: " + str( datetime.datetime.now() ) )
    cell_report_file_name =  "Logs/" + start_date_and_time_string + "/Cell-Report" + ".txt"
    cell_report_file = open( cell_report_file_name, 'w' )
    for line in cell_report_data:
        cell_report_file.write( line )
        cell_report_file.write( "\n" )
    cell_report_file.close()

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

    # Accumulate the active cells and segments and input into report data.
    AccumulateReportData( timeStep, boxColour )

    # Move agents sense organ accordingt to returned vector.
    sensePosX += organVector[ 0 ]
    sensePosY += organVector[ 1 ]
    senseOrgan.goto( sensePosX, sensePosY )

    # Write segment data to individual files for report.
    WriteDataToFiles( timeStep )

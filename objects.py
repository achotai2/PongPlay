import numpy
import atexit
import datetime
import os
import matplotlib.pyplot as plt
from bisect import bisect_left
from cell_and_synapse import BinarySearch

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

stateData             = []
graphY1NumActiveCells = []
graphY2AvgNumSegs     = []
graphY3StabilityScore = []
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

def AccumulateReportData( timeStep, boxColour ):
# Accumulate the active cells and segments and input into report data.

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

    Agent1.SendStateData( stateIndex )

    # Data for graph.
    numActiveCells, averageActiveSegs, stabilityScore = Agent1.GetGraphData()
    graphY1NumActiveCells.append( numActiveCells )
    graphY2AvgNumSegs.append( averageActiveSegs )
    graphY3StabilityScore.append( stabilityScore )
    graphXTimeSteps.append( timeStep )

def exit_handler():
# Upon program exit collects data for Cell-Report log file, and produces the final plot.

    # --------- Add time to log file. ---------
    log_file = open( log_file_name, 'a' )
    log_file.write( "\n" )
    log_file.write( "-------------------------------------------------------" )
    log_file.write( "\n" + "Program End Time: " + str( datetime.datetime.now() ) )
    log_file.close()

    # --------- Prepare Cell-Report and collect individual cell data. ---------
    cell_report_data = []
    cell_report_data.append( start_date_and_time_string )
    cell_report_data.append( "-------------------------------------------------------" )
    cellData = Agent1.GetStateData()

    # Prepare data for state part of report.
    minAcceptablePercentage = 50        # The minimal percent times a cell fires with a state to be considered.

    finalStateCollection = [ [] for i in range( len( stateData ) ) ]
    for cellIdx, cell in enumerate( cellData ):
        for lookingInIndex in range( 1, len( cell ) ):
            stateIdx    = cell[ lookingInIndex ][ 0 ]
            stateCount  = cell[ lookingInIndex ][ 1 ]
            countPct    = int( stateCount * 100 / stateData[ stateIdx ][ 0 ] )
            if countPct > minAcceptablePercentage:
                finalStateCollection[ stateIdx ].append( ( cellIdx, countPct ) )

    for finIdx, item in enumerate( finalStateCollection ):
        cell_report_data.append( "State: " + str( stateData[ finIdx ] ) )
        cell_report_data.append( "Cells Active In State: " + str( sorted( item, key = lambda item: item[ 1 ], reverse = True ) ) )

        itemsCells = [ i[ 0 ] for i in item ]
        for finOverlapIdx, itemOverlap in enumerate( finalStateCollection ):
            if finOverlapIdx != finIdx:
                itemOverlapsCells = [ j[ 0 ] for j in itemOverlap ]
                twoOverlaps       = [ value for value in itemsCells if value in itemOverlapsCells ]
                cell_report_data.append( "States Overlap with State " + str( stateData[ finOverlapIdx ] ) + ": " + str( twoOverlaps ) )
        cell_report_data.append( "" )

    cell_report_data.append( "-------------------------------------------------------" )

    # Prepare data for individual cell part of report.
    cell_report_data.append( "- \t Cell ID \t # Times Active \t ( State active in, % times active ) -" )
    sortedCellIndex = sorted( range( len( cellData ) ), key = lambda k: cellData[ k ], reverse = True )
    for cell in sortedCellIndex:
        state_data_str = ""
        thisCellData = cellData[ cell ]
        entryIndex = 1
        while entryIndex < len( thisCellData ):
            state_data_str += "( " + str( [ stateData[ thisCellData[ entryIndex ][ 0 ] ][ i ] for i in range( 1, len( stateData[ thisCellData[ entryIndex ][ 0 ] ] ) ) ] ) + ", " + str( int( thisCellData[ entryIndex ][ 1 ] / stateData[ thisCellData[ entryIndex ][ 0 ] ][ 0 ] * 100 ) ) + "% ), "
            entryIndex += 1
        cell_report_data.append( "\t" + str( cell ) + "\t\t\t" + str( thisCellData[ 0 ] ) + "\t\t" + state_data_str )

    # Write data into Cell-Report
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
    plt.plot( graphXTimeSteps, graphY2AvgNumSegs, label = "Average # Segments Per Cell" )
    plt.plot( graphXTimeSteps, graphY3StabilityScore, label = "StabilityScore of Working Memory" )
    # naming the x axis
    plt.xlabel('Time Steps')
    # naming the y axis
    plt.ylabel('# of Active')
    # giving a title to my graph
    plt.title('# F-Cells Over Time')
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
    organVector = Agent1.Brain( objCenterX, objCenterY, objWidth, objHeight, boxColour, sensePosX, sensePosY, inputNoisePct )

    # Accumulate the active cells and segments and input into report data.
    AccumulateReportData( timeStep, boxColour )

    # Move agents sense organ accordingt to returned vector.
    sensePosX += organVector[ 0 ]
    sensePosY += organVector[ 1 ]
    senseOrgan.goto( sensePosX, sensePosY )

    # Write segment data to individual files for report.
    WriteDataToFiles( timeStep )

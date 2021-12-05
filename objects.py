import numpy
import atexit
import datetime
import os

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

atexit.register( exit_handler )

# Main program loop-------------------------------------------------------------
while True:

    wn.update()         # Screen update

    timeStep += 1

#    if int( timeStep / 100 ) % 3 == 1:
#        boxColour = 2
#        box.color( "green" )
#    elif int( timeStep / 100 ) % 3 == 2:
#        boxColour = 3
#        box.color( "blue" )
#    elif int( timeStep / 100 ) % 3 == 0:
#        boxColour = 1
#        box.color( "red" )

    # Run agent brain and get motor vector.
    organVector = Agent1.Brain( objCenterX, objCenterY, objWidth, objHeight, boxColour, sensePosX, sensePosY )

    # Move agents sense organ accordingt to returned vector.
    sensePosX += organVector[ 0 ]
    sensePosY += organVector[ 1 ]
    senseOrgan.goto( sensePosX, sensePosY )

    # Write segment data to individual files for report.
    WriteDataToFiles( timeStep )

import datetime
import os
import matplotlib.pyplot as plt

class AgentLog:

    def __init__( self, name, startTimeString ):
    # Set up log data for each agent.

        self.name            = name
        self.startTimeString = startTimeString

        self.log_file_name = "Logs/" + startTimeString + "/Log_" + str( name ) + "_Agent.txt"        # Stores the agents file names.
        log_file = open( self.log_file_name, 'a' )
        log_file.write( "Agent " + str( name ) )
        log_file.write( "Program Start Time: " + str( startTimeString ) )
        log_file.write( "\n" )
        log_file.close()

        self.fcell_report_file_name = "Logs/" + startTimeString + "/F-Cell-Report_" + str( name ) + "_Agent.txt"
#        self.ocell_report_file_name = "Logs/" + startTimeString + "/O-Cell-Report_" + str( name ) + "_Agent.txt"

        self.graphY1NumActiveCells = []
        self.graphY2NumSegs        = []
        self.graphY3NumPredicCells = []
        self.graphXTimeSteps       = []

    def WriteToLogFile( self, new_data, timeStep, reflecting ):
    # Write sent list of data to file.

        # Create text file to store log data.
        log_data = []
        log_data.append( "-------------------------------------------------------" )
        log_data.append( "Time Step: " + str( timeStep ) )

        if reflecting:
            log_data.append( "Reflecting..." )
        else:
            log_data.append( "Computing..." )

        for entry in new_data:
            log_data.append( str( entry ) )

        # Write log data to text file.
        log_file = open( self.log_file_name, 'a' )
        for line in log_data:
            log_file.write( line )
            log_file.write( "\n" )
        log_file.close()

    def FeedForGraphs( self, numActiveCells, numActiveSegs, numPredictedCells, timeStep, reflecting ):
    # Feed data for graphs.

        if not reflecting:
            self.graphY1NumActiveCells.append( numActiveCells )
            self.graphY2NumSegs.append( numActiveSegs )
            self.graphY3NumPredicCells.append( numPredictedCells )
            self.graphXTimeSteps.append( timeStep )
        else:
            self.graphY1NumActiveCells.append( 0 )
            self.graphY2NumSegs.append( 0 )
            self.graphY3NumPredicCells.append( 0 )
            self.graphXTimeSteps.append( timeStep )

    def EndLog( self, endTime ):
    # Enter the end time stamp for log.

        log_file = open( self.log_file_name, 'a' )
        log_file.write( "\n" )
        log_file.write( "-------------------------------------------------------" )
        log_file.write( "\n" + "Program End Time: " + str( endTime ) )
        log_file.close()

    def FCellReport( self, fCellData, stateData, endTime, minAcceptablePercentage ):
    # Prepare and write the FCell report log.

        fcell_report_data = []

        fcell_report_data.append( self.startTimeString )
        fcell_report_data.append( "Agent: " + str( self.name ) )
        fcell_report_data.append( "-------------------------------------------------------" )

        # Prepare data for state part of report.
        finalStateCollection = [ [] for i in range( len( stateData ) ) ]
        for cellIdx, cell in enumerate( fCellData ):
            for lookingInIndex in range( 1, len( cell ) ):
                stateIdx    = cell[ lookingInIndex ][ 0 ]
                stateCount  = cell[ lookingInIndex ][ 1 ]
                countPct    = int( stateCount * 100 / stateData[ stateIdx ][ 0 ] )
                if countPct > minAcceptablePercentage:
                    finalStateCollection[ stateIdx ].append( ( cellIdx, countPct ) )

        for finIdx, item in enumerate( finalStateCollection ):
            fcell_report_data.append( "State: " + str( stateData[ finIdx ] ) )
            fcell_report_data.append( "Cells Active In State: " + str( len( item ) ) + ", " + str( sorted( item, key = lambda item: item[ 1 ], reverse = True ) ) )

            itemsCells = [ i[ 0 ] for i in item ]
            for finOverlapIdx, itemOverlap in enumerate( finalStateCollection ):
                if finOverlapIdx != finIdx:
                    itemOverlapsCells = [ j[ 0 ] for j in itemOverlap ]
                    twoOverlaps       = [ value for value in itemsCells if value in itemOverlapsCells ]
                    fcell_report_data.append( "Overlap with State " + str( stateData[ finOverlapIdx ] ) + ": " + str( len( twoOverlaps ) ) + ", " + str( twoOverlaps ) )
            fcell_report_data.append( "" )

        fcell_report_data.append( "-------------------------------------------------------" )

        # Prepare data for individual cell part of report.
        fcell_report_data.append( "- \t Cell ID \t # Times Active \t ( State active in, % times active ) -" )
        sortedCellIndex = sorted( range( len( fCellData ) ), key = lambda k: fCellData[ k ], reverse = True )
        for cell in sortedCellIndex:
            state_data_str = ""
            thisfCellData = fCellData[ cell ]
            entryIndex = 1
            while entryIndex < len( thisfCellData ):
                state_data_str += "( " + str( [ stateData[ thisfCellData[ entryIndex ][ 0 ] ][ i ] for i in range( 1, len( stateData[ thisfCellData[ entryIndex ][ 0 ] ] ) ) ] ) + ", " + str( int( thisfCellData[ entryIndex ][ 1 ] / stateData[ thisfCellData[ entryIndex ][ 0 ] ][ 0 ] * 100 ) ) + "% ), "
                entryIndex += 1
            fcell_report_data.append( "\t" + str( cell ) + "\t\t\t" + str( thisfCellData[ 0 ] ) + "\t\t" + state_data_str )

        # Write data into F-Cell-Report
        fcell_report_data.append( "\n" + "Program End Time: " + str( endTime ) )
        fcell_report_file = open( self.fcell_report_file_name, 'w' )
        for line in fcell_report_data:
            fcell_report_file.write( line )
            fcell_report_file.write( "\n" )
        fcell_report_file.close()

#    def OCellReport( self, oCellData, stateData, endTime ):
#    # Prepare and write the OCell report log.
#
#        ocell_report_data = []
#
#        ocell_report_data.append( self.startTimeString )
#        ocell_report_data.append( "Agent: " + str( self.name ) )
#        ocell_report_data.append( "-------------------------------------------------------" )
#
#        # Prepare the OCell Report data.
#        oCellData.sort( key = lambda oCellData: oCellData[ 0 ] )
#        for entryIndex, entry in enumerate( oCellData ):
#            ocell_report_data.append( str( entry[ 0 ] ) + ": " + str( entry[ 1 ] ) )
#
#        # Write data into O-Cell-Report
#        ocell_report_data.append( "\n" + "Program End Time: " + str( endTime ) )
#        ocell_report_file = open( self.ocell_report_file_name, 'w' )
#        for line in ocell_report_data:
#            ocell_report_file.write( line )
#            ocell_report_file.write( "\n" )
#        ocell_report_file.close()

    def PlotGraphs( self ):
    # Plot the graphs.

        # plotting the points
#        plt.plot( graphXTimeSteps, graphY2NumSegs, label = "# Active Segments" )
        plt.plot( self.graphXTimeSteps, self.graphY3NumPredicCells, label = "# Predicted F-Cells" )
        plt.plot( self.graphXTimeSteps, self.graphY1NumActiveCells, label = "# Active F-Cells" )
        # naming the x axis
        plt.xlabel( 'Time Steps' )
        # naming the y axis
        plt.ylabel( '# of Active' )
        # giving a title to my graph
        plt.title( 'Data Over Time' )
        plt.legend()
        # function to show the plot
        plt.savefig( "Logs/" + self.startTimeString + "/Plot_Data_" + str( self.name ) + "_Agent.png" )
        plt.show()

class Logging:

    minAcceptablePercentage = 50        # The minimal percent times a cell fires with a state to be considered.

    def __init__( self, agentsNamesList ):

        self.timeStep  = 0
        self.stateData = []                     # Stores the various states, and their occurrance count.

        # Prepare log and report files and folders------------------------------
        self.start_date_and_time_string = str( datetime.datetime.now() )
        os.mkdir( "./Logs/" + self.start_date_and_time_string )                  # Create logging folder.

        self.agentsLogs = []
        for thisName in agentsNamesList:
            self.agentsLogs.append( AgentLog( thisName, self.start_date_and_time_string ) )

    def AddToTimeStep( self ):
    # Just add one to timeStep.

        self.timeStep += 1

    def WriteDataToFiles( self, AgentList ):
    # Go through all returned segment data and write it to separate segment files.

        for aIdx, Agent in enumerate( AgentList ):
            new_data, reflecting = Agent.GetLogData()

            self.agentsLogs[ aIdx ].WriteToLogFile( new_data, self.timeStep, reflecting )

    def AccumulateReportData( self, AgentList, currState ):
    # Adds currState to stateData, which stores all states and the number of times that state has occurred.
    # Accumulate the active cells and segments and input into report data.

        if len( self.stateData ) > 0 and len( currState ) != len( self.stateData[ 0 ] ) - 1:
            print( "State information sent to AccumulateReportData() not of right size." )
            exit()

        # Gather state data.
        stateIndex    = 0
        while stateIndex < len( self.stateData ):
            if all( [ self.stateData[ stateIndex ][ i + 1 ] == currState[ i ] for i in range( len( currState ) ) ] ):
                self.stateData[ stateIndex ][ 0 ] += 1
                break
            stateIndex += 1

        if stateIndex == len( self.stateData ):
            newEntry = [ 1 ]                        # The first index in entries in stateData is the occurrance count.
            for entry in currState:
                newEntry.append( entry )
            self.stateData.append( newEntry )

        # Send the state to each Agent for their records.
        for aIdx, Agent in enumerate( AgentList ):
            Agent.SendStateData( stateIndex )

            # Data for graph.
            cellData, reflecting = Agent.GetGraphData()
            numActiveCells    = cellData[ 0 ]
            numActiveSegs     = cellData[ 1 ]
            numPredictedCells = cellData[ 2 ]
            self.agentsLogs[ aIdx ].FeedForGraphs( numActiveCells, numActiveSegs, numPredictedCells, self.timeStep, reflecting )

    def WhenExit( self, AgentList ):
    # Finalize log files and prepare graphs.

        endTime = datetime.datetime.now()

        for aIdx, Agent in enumerate( AgentList ):
            # Add time to log files.
            self.agentsLogs[ aIdx ].EndLog( datetime.datetime.now() )

            # Prepare Cell-Report and collect individual cell data.
#            fCellData, oCellData = Agent.GetStateData()
            fCellData = Agent.GetStateData()

            self.agentsLogs[ aIdx ].FCellReport( fCellData, self.stateData, endTime, self.minAcceptablePercentage )

#            self.agentsLogs[ aIdx ].OCellReport( oCellData, self.stateData, endTime )

            # Plot the graphs.
            self.agentsLogs[ aIdx ].PlotGraphs()

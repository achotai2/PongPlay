from random import sample, randrange
from operator import add
from cell_and_synapse import FCell, OCell, BinarySearch, IndexIfItsIn, NoRepeatInsort, RepeatInsort, CheckInside
import numpy as np
from time import time

class VectorMemory:

    def __init__( self, columnDimensions, cellsPerColumn, numObjectCells, FActivationThresholdMin, FActivationThresholdMax, initialPermanence, lowerThreshold,
        permanenceIncrement, permanenceDecrement, permanenceDecay, segmentDecay, initialPosVariance, ObjectRepActivaton, OActivationThreshold,
        maxSynapsesToAddPer, maxSegmentsPerCell, maxSynapsesPerSegment, equalityThreshold, pctAllowedOCellConns ):

        self.columnDimensions        = columnDimensions         # Dimensions of the column space.
        self.FCellsPerColumn         = cellsPerColumn           # Number of cells per column.
        self.numObjectCells          = numObjectCells           # Number of cells in the Object level.
        self.FActivationThresholdMin = FActivationThresholdMin  # Min threshold of active connected incident synapses...
        self.FActivationThresholdMax = FActivationThresholdMax  # Max threshold of active connected incident synapses...
                                                                # needed to activate segment.
        self.initialPermanence       = initialPermanence        # Initial permanence of a new synapse.
        self.lowerThreshold          = lowerThreshold           # The lowest permanence for synapse to be active.
        self.permanenceIncrement     = permanenceIncrement      # Amount by which permanences of synapses are incremented during learning.
        self.permanenceDecrement     = permanenceDecrement      # Amount by which permanences of synapses are decremented during learning.
        self.permanenceDecay         = permanenceDecay          # Amount to decay permances each time step if < 1.0.
        self.segmentDecay            = segmentDecay             # If a segment hasn't been active in this many time steps then delete it.
        self.initialPosVariance      = initialPosVariance       # Amount of range vector positions are valid in.
        self.ObjectRepActivaton      = ObjectRepActivaton       # Number of active OCells in object layer at one time.
        self.OActivationThreshold    = OActivationThreshold     # Threshold of active connected OToFSynapses...
                                                                # needed to activate OCell.
        self.maxSynapsesToAddPer      = maxSynapsesToAddPer       # The maximum number of FToFSynapses added to a segment during creation.
        self.maxSegmentsPerCell      = maxSegmentsPerCell       # The maximum number of segments per cell.
        self.maxSynapsesPerSegment   = maxSynapsesPerSegment     # Maximum number of active synapses allowed on a segment.
        self.equalityThreshold       = equalityThreshold        # The number of equal synapses for two segments to be considered identical.
        self.pctAllowedOCellConns    = pctAllowedOCellConns     # Percent of OCells an FCell can build connections to.

        # --------Create all the cells in the network.---------

        self.columnSDR    = []
        self.burstingCols = []

        # Create cells in feature layer.
        self.FCells = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell( initialPermanence, numObjectCells, pctAllowedOCellConns, segmentDecay ) )

        # Create cells in the object layer.
        self.OCells = []
        for i in range( numObjectCells ):
            self.OCells.append( OCell() )

        self.lastActiveColumns = []

        self.workingMemory = []

    def SendData( self, stateNumber ):
    # Return the active FCells as a list.

        for fCell in self.FCells:
            fCell.ReceiveStateData( stateNumber )

    def GetStateInformation( self ):
    # Get all the cells state information and return it.

        toReturn = []
        for fCell in self.FCells:
            toReturn.append( fCell.ReturnStateInformation() )

        return toReturn

    def GetGraphData( self ):
    # Return the number of active cells.

        numActiveCells = 0
        sumSegs        = 0

        for cell in self.FCells:
            if cell.active:
                numActiveCells += 1
                sumSegs += len( cell.segments )

        return numActiveCells, int( sumSegs / numActiveCells )

    def BuildLogData( self, log_data ):
    # Adds important information to log_data for entry into log.

        log_data.append( "Active Columns: " + str( len( self.columnSDR ) ) + ", " + str( self.columnSDR ) )

        activeFCells = []
        for index, fCell in enumerate( self.FCells ):
            if fCell.active:
                activeFCells.append( index )
        log_data.append( "Active F-Cells: " + str( len( activeFCells ) ) + ", " + str( activeFCells ) )

        log_data.append( "Bursting Column Pct: " + str( len( self.burstingCols ) / self.columnDimensions * 100 ) + "%, " + str( self.burstingCols ) )

        activeOCells = []
        for index, oCell in enumerate( self.OCells ):
            if oCell.active:
                activeOCells.append( index )
        log_data.append( "Active O-Cells: " + str( len( activeOCells ) ) + ", " + str( activeOCells ) )

        predictedFCells = []
        for index, fCell in enumerate( self.FCells ):
            if fCell.predicted:
                predictedFCells.append( index )
        log_data.append( "Predicted Cells: " + str( len( predictedFCells ) ) + ", " + str( predictedFCells ) )

        log_data.append( "Working Memory: " + str( self.workingMemory ) )

    def ActivateFCells( self ):
    # Uses activated columns and cells in predicted state to put cells in active states.
    # Return a list of winnerCells.

        # Clean up old active and lastActive FCells.
        for fCell in self.FCells:
            if fCell.lastActive:
                fCell.lastActive = False
                fCell.lastWinner = False

            if fCell.active:
                fCell.active     = False
                fCell.lastActive = True
                if fCell.winner:
                    fCell.winner     = False
                    fCell.lastWinner = True

        self.burstingCols    = []
        predictedActiveCells = []

        for col in self.columnSDR:
            predictedCellsThisCol = []

            # Check if any cells in column are predicted. If yes then make them active.
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cell ].predicted:
                    predictedCellsThisCol.append( cell )

            if len( predictedCellsThisCol ) > 0:
                theCell = predictedCellsThisCol[ 0 ]
                for predCell in predictedCellsThisCol:
                    if self.FCells[ predCell ].HighestOverlapForActiveSegment() > self.FCells[ theCell ].HighestOverlapForActiveSegment():
                        theCell = predCell

                self.FCells[ theCell ].active = True
                self.FCells[ theCell ].winner = True
                predictedActiveCells.append( theCell )

            # If none predicted then burst column, making all cells in column active.
            # Choose a winner cell at random.
            else:
                self.burstingCols.append( col )

                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    self.FCells[ cell ].active = True

                self.FCells[ randrange( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ) ].winner = True

        return predictedActiveCells

    def ActivateOCells( self, predictedActiveCells ):
    # Use the active FCells to activate the OCells.

        if len( predictedActiveCells ) > self.FActivationThresholdMax:
# THIS WILL NEED CLEANING UP LATER. WE ARE BASICALLY TAKING THE CELLS WE ALREADY HAVE AND COMPILING A NEW LIST FOR
# DecayAndCreate TO USE IN BOOL FORM.
            # Generate the predicted active FCells in bool form.
            predictedActiveCellsBool = [ False ] * self.FCellsPerColumn * self.columnDimensions
            for fCell in predictedActiveCells:
                predictedActiveCellsBool[ fCell ] = True

            # Count the number of active OCells
            numActiveOCells = 0
            for oCell in self.OCells:
                if oCell.active:
                    numActiveOCells += 1
                # Check all inactive OCells's segments overlap for predictedActiveCells and see if they become active.
                else:
                    if oCell.CheckOverlapAndActivate( predictedActiveCellsBool, self.FCellsPerColumn, self.lowerThreshold ):
                        numActiveOCells += 1
            
            # If there are still not enough active OCells then activate random ones.
            while numActiveOCells < self.ObjectRepActivaton:
                randomOn = randrange( 0, self.numObjectCells )
                if not self.OCells[ randomOn ].active:
                    self.OCells[ randomOn ].active = True
                    numActiveOCells += 1

            # If there are too many active OCells then deactivate the ones with least overlap score.
            if numActiveOCells > self.ObjectRepActivaton:
                activeOCells = []
                for index, aOCell in enumerate( self.OCells ):
                    if aOCell.active:
                        activeOCells.append( ( index, aOCell.ReturnGreatestOverlapScore() ) )

                activeOCells.sort( key = lambda activeOCells: activeOCells[ 1 ] )
                while numActiveOCells > self.ObjectRepActivaton:
                    self.OCells[ activeOCells.pop( 0 )[ 0 ] ].Deactivate()
                    numActiveOCells -= 1

            # Check all active OCells if they have segments which activate with given predictedActiveCells.
            # If they do then activate and strengthen segment, perform learning on it, DecayAndCreate.
            # If they don't then create a new segment.
            for oCell in self.OCells:
                if oCell.active:
                    oCell.OCellLearning( predictedActiveCells, predictedActiveCellsBool, self.FCellsPerColumn, self.lowerThreshold, self.permanenceIncrement, self.permanenceDecrement, self.initialPermanence, self.maxSynapsesToAddPer, self.maxSynapsesPerSegment, self.FActivationThresholdMin )
                    numActiveOCells += 1

        else:
            # If there not enough predicted active cells (current feature wasn't predicted), then deactivate all OCells.
            for oCell in self.OCells:
                oCell.Deactivate()

    def OCellLearning( self ):
    # Using the currently active and last active FCells, build stronger synapses to the active OCells,
    # and weaken to the losers.

        # Add all the active and last active OCells.
        allActiveOCells = []
        for actOCell in self.activeOCells:
            NoRepeatInsort( allActiveOCells, actOCell )
        for lActOCell in self.lastActiveOCells:
            NoRepeatInsort( allActiveOCells, lActOCell )

        # For all active and lastactive FCells strengthen to the active and lastActive OCells, and weaken to others.
        for actOCell in self.activeOCells:
            self.FCells[ actOCell ].OCellConnect( allActiveOCells, self.permanenceIncrement, self.permanenceDecrement )
        for lActOCell in self.lastActiveOCells:
            self.FCells[ lActOCell ].OCellConnect( allActiveOCells, self.permanenceIncrement, self.permanenceDecrement )

    def UpdateWorkingMemory( self, vector ):
    # Use the vector to update all vectors stored in workingMemory items, and add timeStep. If any
    # item is above threshold time steps then remove it.
    # Add the new active Fcells to working memory.

        if len( self.workingMemory ) > 0:

            workingMemoryToDelete = []

            for index in range( len( self.workingMemory ) ):
                # Update vector.
                self.workingMemory[ index ][ 1 ][ 0 ] += vector[ 0 ]
                self.workingMemory[ index ][ 1 ][ 1 ] += vector[ 1 ]

# WILL HAVE TO WORK ON WORKING MEMORY TO MAKE IT MORE PRECISE.
                if self.workingMemory[ index ][ 1 ][ 0 ] == 0 and self.workingMemory[ index ][ 1 ][ 1 ] == 0:
                    workingMemoryToDelete.insert( 0, index )

                # Update time step.
                self.workingMemory[ index ][ 2 ] += 1

                if self.workingMemory[ index ][ 2 ] > self.segmentDecay:
                    if workingMemoryToDelete[ 0 ] != index:
                        workingMemoryToDelete.insert( 0, index )

            # Delete items whose timeStep is above threshold.
            if len( workingMemoryToDelete ) > 0:
                for toDel in workingMemoryToDelete:
                    del self.workingMemory[ toDel ]

        # Add active winner cells to working memory, with zero vector (since we're there), and no time steps,
        # if it doesn't exist.
        winnerCells = []
        for index, cell in enumerate( self.FCells ):
            if cell.winner:
                winnerCells.append( index )
        self.workingMemory.append( [ winnerCells, [ 0, 0 ], 0 ] )

    def PredictFCells( self, vector ):
    # Clear old predicted FCells and generate new predicted FCells.
        # Get a bool list of all active and not active FCells.
        activeFCellsBool = []
        for fCell in self.FCells:
            activeFCellsBool.append( fCell.active )

        # Check every activeFCell's segments, and activate or deactivate them; make FCell predicted or not.
        for cell in self.FCells:

            # Make previously active segments lastActive (used in segment learning), and lastActive off.
            cell.UpdateSegmentActivity()

            # Check the cell against ative cells to see if any segment predicts this cell as terminal.
            cell.CheckIfPredicted( activeFCellsBool, self.FCellsPerColumn, vector, self.lowerThreshold )

        # If no segments are predicting cells then look into working memory if this vector is predicted there.
        wokePredicted = False
        if len( self.workingMemory ) > 0:
            for item in self.workingMemory:
                if CheckInside( [ 0, 0 ], list( map( add, item[1], vector ) ), self.initialPosVariance ):
                    for cell in item[ 0 ]:
                        wokePredicted = True
                        self.FCells[ cell ].predicted = True

    def Compute( self, columnSDR, lastVector ):
    # Compute the action of vector memory, and learn on the synapses.

        self.columnSDR = columnSDR.sparse.tolist()

        # Safety check for column dimensions.
        if columnSDR.size != self.columnDimensions:
            print( "VM input column dimensions must be same as input SDR dimensions." )
            exit()

        # Clear old active cells and get new ones active cells for this time step.
        predictedActiveCells = self.ActivateFCells()

        # Compile lists of last active FCells and lastActive winner FCells, for use below.
        lastActiveCellsBool = []
        lastWinnerCells     = []
        for index, fCell in enumerate( self.FCells ):
            lastActiveCellsBool.append( fCell.lastActive )
            if fCell.lastWinner:
                lastWinnerCells.append( index )

        # Update working memory vectors.
        self.UpdateWorkingMemory( lastVector )

        # Use the active FCells to tell us what OCells to activate.
#        self.ActivateOCells( predictedActiveCells )

#        # Make sure that there is no more than one winner per column in incidentCellWinners.
#        # If there is more then ignore that column, for creating new synapses, by deleting it from incidentCellWinners.
#        lastWinnerCells     = []
#        for col in self.lastActiveColumns:
#            numLWinners = 0
#            for cell in range( self.FCellsPerColumn ):
#                if self.FCells[ ( col * self.FCellsPerColumn ) + cell ].lastWinner:
#                    chosenCell = ( col * self.FCellsPerColumn ) + cell
#                    numLWinners += 1
#
#            if numLWinners == 1:
#                lastWinnerCells.append( chosenCell )

        for cell in self.FCells:
            # Perform learning on Segments in FCells.
            cell.SegmentLearning( lastVector, lastActiveCellsBool, lastWinnerCells, self.initialPermanence, self.initialPosVariance,
                self.FActivationThresholdMin, self.FActivationThresholdMax, self.permanenceIncrement, self.permanenceDecrement,
                self.maxSynapsesToAddPer, self.maxSynapsesPerSegment, self.FCellsPerColumn, self.equalityThreshold )

#            self.OCellLearning()

        self.lastActiveColumns = self.columnSDR

        return None

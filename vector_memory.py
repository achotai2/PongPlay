from random import sample, randrange
from cell_and_synapse import FCell, BinarySearch, IndexIfItsIn, NoRepeatInsort, RepeatInsort, CheckInside
import numpy as np
from time import time

class VectorMemory:

    def __init__( self, columnDimensions, cellsPerColumn, numObjectCells, FActivationThresholdMin, FActivationThresholdMax, initialPermanence, lowerThreshold,
        permanenceIncrement, permanenceDecrement, permanenceDecay, segmentDecay, initialPosVariance, ObjectRepActivaton, OActivationThreshold,
        maxNewFToFSynapses, maxSegmentsPerCell, maxSynapsesPerSegment, equalityThreshold, pctAllowedOCellConns ):

        self.columnDimensions        = columnDimensions         # Dimensions of the column space.
        self.FCellsPerColumn         = cellsPerColumn           # Number of cells per column.
        self.numObjectCells          = numObjectCells           # Number of cells in the Object level.
        self.FActivationThresholdMin = FActivationThresholdMin  # Threshold of active connected incident synapses...
                                                                # needed to activate segment.
        self.FActivationThresholdMax = FActivationThresholdMax  # Threshold of active connected incident synapses...
        self.initialPermanence       = initialPermanence        # Initial permanence of a new synapse.
        self.lowerThreshold          = lowerThreshold           # The lowest permanence for synapse to be active.
        self.permanenceIncrement     = permanenceIncrement      # Amount by which permanences of synapses are incremented during learning.
        self.permanenceDecrement     = permanenceDecrement      # Amount by which permanences of synapses are decremented during learning.
        self.permanenceDecay         = permanenceDecay          # Amount to decay permances each time step if < 1.0.
        self.segmentDecay            = segmentDecay             # If a segment hasn't been active in this many time steps then delete it.
        self.initialPosVariance      = initialPosVariance       # Amount of range vector positions are valid in.
        self.ObjectRepActivaton      = ObjectRepActivaton       # NNumber of active OCells in object layer at one time.
        self.OActivationThreshold    = OActivationThreshold     # Threshold of active connected OToFSynapses...
                                                                # needed to activate OCell.
        self.maxNewFToFSynapses      = maxNewFToFSynapses       # The maximum number of FToFSynapses added to a segment during creation.
        self.maxSegmentsPerCell      = maxSegmentsPerCell       # The maximum number of segments per cell.
        self.maxSynapsesPerSegment   = maxSynapsesPerSegment     # Maximum number of active bundles allowed on a segment.
        self.equalityThreshold       = equalityThreshold        # The number of equal synapses for two segments to be considered identical.
        self.pctAllowedOCellConns    = pctAllowedOCellConns     # Percent of OCells an FCell can build connections to.

        # --------Create all the cells in the network.---------

        self.columnSDR    = []
        self.burstingCols = []

        # Create cells in feature layer.
        self.FCells            = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell( initialPermanence, numObjectCells, pctAllowedOCellConns, segmentDecay ) )

        self.lastActiveColumns = []

        # Create cells in object layer.
        self.activeOCells     = []
        self.lastActiveOCells = []

        self.workingMemory = []

    def ReturnData( self ):
    # Return the active FCells as a list.

        activeFCells = []
        for fCell in self.FCells:
            activeFCells.append( fCell.active )

        return activeFCells

    def BuildLogData( self, log_data ):
    # Adds important information to log_data for entry into log.

        log_data.append( "Active Columns: " + str( len( self.columnSDR ) ) + ", " + str( self.columnSDR ) )

        activeFCells = []
        for index, fCell in enumerate( self.FCells ):
            if fCell.active:
                activeFCells.append( index )
        log_data.append( "Active F-Cells: " + str( len( activeFCells ) ) + ", " + str( activeFCells ) )

        log_data.append( "Bursting Column Pct: " + str( len( self.burstingCols ) / self.columnDimensions * 100 ) + "%, " + str( self.burstingCols ) )
#        log_data.append( "Active O-Cells: " + str( self.activeOCells ) )

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

        self.burstingCols = []

        for col in self.columnSDR:
            columnpredicted = False

            # Check if any cells in column are predicted. If yes then make them active.
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cell ].predicted:
                    columnpredicted = True
                    self.FCells[ cell ].active = True
                    self.FCells[ cell ].winner = True

            # If none predicted then burst column, making all cells in column active.
            # Make the one with least chosen winners the winner cell.
            if not columnpredicted:
                self.burstingCols.append( col )

                minSegmentsNum = -1
                minSegmentsIdx = 0
                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    self.FCells[ cell ].active = True

                    if minSegmentsNum == -1 or self.FCells[ cell ].NumSegments() < minSegmentsNum:
                        minSegmentsNum = self.FCells[ cell ].NumSegments()
                        minSegmentsIdx = cell

                self.FCells[ minSegmentsIdx ].winner = True

    def ActivateOCells( self ):
    # Use the active FCells to activate the OCells.

        # Refresh old active OCells.
        self.lastActiveOCells = self.activeOCells.copy()
        self.activeOCells = []

        oCellActivationLevel = [ 0 ] * self.numObjectCells
        # Use active FCells to decide on new active OCells.
        for index, fCell in enumerate( self.FCells ):
            if fCell.active:
                for index, oCell in enumerate( fCell.OCellConnections ):
                    if fCell.OCellPermanences[ index ] > self.lowerThreshold:
                        oCellActivationLevel[ oCell ] += 1

#        potentialActiveOcells = []
#        for index, oCell in enumerate( oCellActivationLevel ):
#            if oCell > self.OActivationThreshold:
#                potentialActiveOcells.append( ( oCell, index ) )
#        potentialActiveOcells.sort( reverse = True )

#        for a in range( self.ObjectRepActivaton ):
#            NoRepeatInsort( self.activeOCells, potentialActiveOcells[ a ][ 1 ] )

        for index, oCell in enumerate( oCellActivationLevel ):
            if oCell > self.OActivationThreshold:
                NoRepeatInsort( self.activeOCells, index )

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

        # Also look into working memory if this vector is predicted there.
        if len( self.workingMemory ) > 0:
            for item in self.workingMemory:
                if CheckInside( item[ 1 ], vector, self.initialPosVariance ):
                    for fCell in item[ 0 ]:
                        self.FCells[ fCell ].predicted = True

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

    def Compute( self, columnSDR, lastVector, newVector ):
    # Compute the action of vector memory, and learn on the synapses.

        self.columnSDR = columnSDR.sparse.tolist()

        # Safety check for column dimensions.
        if columnSDR.size != self.columnDimensions:
            print( "VM input column dimensions but be same as input SDR dimensions." )
            exit()

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells()

        # Update working memory vectors.
        self.UpdateWorkingMemory( lastVector )

        # Use the active FCells to tell us what OCells to activate.
        self.ActivateOCells()

        # Perform learning on Segments in FCells.
        lastWinnerCells     = []
        lastActiveCellsBool = []
        for index, fCell in enumerate( self.FCells ):
            lastActiveCellsBool.append( fCell.lastActive )
            if fCell.lastWinner:
                lastWinnerCells.append( index )

        for cell in self.FCells:

            cell.SegmentLearning( lastVector, lastActiveCellsBool, lastWinnerCells, self.initialPermanence,
                self.initialPosVariance, self.FActivationThresholdMin, self.FActivationThresholdMax, self.permanenceIncrement,
                self.permanenceDecrement, self.maxNewFToFSynapses, self.maxSynapsesPerSegment, self.FCellsPerColumn )

#            self.OCellLearning()

        # Use FSegments to predict next set of inputs, given newVector.
        self.PredictFCells( newVector )

        self.lastActiveColumns = self.columnSDR

from random import sample, randrange
from operator import add
from cell_and_synapse import FCell, OCell, BinarySearch, IndexIfItsIn, NoRepeatInsort, RepeatInsort, CheckInside
import numpy as np

class VectorMemory:

    def __init__( self, columnDimensions, cellsPerColumn, numObjectCells, FActivationThresholdMin, FActivationThresholdMax, initialPermanence, lowerThreshold,
        permanenceIncrement, permanenceDecrement, permanenceDecay, segmentDecay, initialPosVariance, ObjectRepActivaton, OActivationThreshold,
        maxSynapsesToAddPer, maxSegmentsPerCell, maxSynapsesPerSegment, equalityThreshold, pctExcitationActivation ):

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
        self.maxSynapsesToAddPer     = maxSynapsesToAddPer       # The maximum number of FToFSynapses added to a segment during creation.
        self.maxSegmentsPerCell      = maxSegmentsPerCell       # The maximum number of segments per cell.
        self.maxSynapsesPerSegment   = maxSynapsesPerSegment     # Maximum number of active synapses allowed on a segment.
        self.equalityThreshold       = equalityThreshold        # The number of equal synapses for two segments to be considered identical.
        self.pctExcitationActivation = pctExcitationActivation  # Percent of excited segments for OCell to become active.

        # --------Create all the cells in the network.---------

        self.columnSDR       = []
        self.burstingCols    = []
        self.activeFCells    = []
        self.winnerFCells    = []
        self.predictedFCells = []

        # Create cells in feature layer.
        self.FCells = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell( initialPermanence, numObjectCells ) )

        # Create cells in the object layer.
        self.OCells = []
        for i in range( numObjectCells ):
            self.OCells.append( OCell() )
        # Activate a set of random OCells to begin with.
        self.activeOCells = sample( range( numObjectCells ), ObjectRepActivaton )
        for j in self.activeOCells:
            self.OCells[ j ].active  = True

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

        log_data.append( "Active F-Cells: " + str( len( self.activeFCells ) ) + ", " + str( self.activeFCells ) )

        log_data.append( "Bursting Column Pct: " + str( len( self.burstingCols ) / self.columnDimensions * 100 ) + "%, " + str( self.burstingCols ) )

        log_data.append( "Active O-Cells: " + str( len( self.activeOCells ) ) + ", " + str( self.activeOCells ) )

        log_data.append( "Predicted Cells: " + str( len( self.predictedFCells ) ) + ", " + str( self.predictedFCells ) )

    def ActivateFCells( self ):
    # Uses activated columns and cells in predicted state to put cells in active states.
    # Return a list of winnerCells.

        # Clean up old active and lastActive FCells.
# WE MIGHT BE ABLE TO Get RID OF ALL THE LASTACTIVE, LASTWINNER. IS IT USEFUL IN THE NEW SYSTEM?
        for fCell in self.FCells:
            fCell.UpdateCellState()
        self.activeFCells = []
        self.winnerFCells = []
        self.burstingCols = []

        for col in self.columnSDR:
            predictedCellsThisCol = []

            # Check if any cells in column are predicted. If yes then make them active.
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cell ].predicted:
                    predictedCellsThisCol.append( cell )

            if len( predictedCellsThisCol ) > 0:
                theCell = predictedCellsThisCol[ 0 ]
                for predCell in predictedCellsThisCol:
# THIS IS CHANGING IN THE CURRENT SYSTEM BECAUSE THE FCELLS DONT HAVE SEGMENTS
                    if self.FCells[ predCell ].HighestOverlapForActiveSegment() > self.FCells[ theCell ].HighestOverlapForActiveSegment():
                        theCell = predCell

                self.FCells[ theCell ].active = True
                NoRepeatInsort( self.activeFCells, theCell )
                self.FCells[ theCell ].winner = True
                NoRepeatInsort( self.winnerFCells, theCell )

            # If none predicted then burst column, making all cells in column active.
            # Choose a winner cell at random.
            else:
                self.burstingCols.append( col )

                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    self.FCells[ cell ].active = True
                    NoRepeatInsort( self.activeFCells, cell )

                theWinner = randrange( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn )
                self.FCells[ theWinner ].winner = True
                NoRepeatInsort( self.winnerFCells, theWinner )

        for fCell in self.FCells:
# SHOULD EVENTUALLY MOVE THIS TO AFTER THE OCELLS BECOME ACTIVE, ONCE WE CHANGE OCELL ACTIVATION.
            fCell.SegmentLearning( self.activeOCells, self.initialPermanence, self.FActivationThresholdMin,
                self.FActivationThresholdMax, self.segmentDecay, self.permanenceIncrement, self.permanenceDecrement,
                self.maxSynapsesToAddPer, self.maxSynapsesPerSegment, self.FCellsPerColumn, self.equalityThreshold )

    def ActivateOCells( self, lastVector ):
    # Use the active FCells to activate the OCells.

        for oCell in self.OCells:

            # Check if oCell becomes active.
#            oCell.CheckOCellActivation( self.pctExcitationActivation )

            # Update activation on all segments.
            oCell.UpdateSegmentActivity()

            # Update the vector hypothesis on this OCell.
            oCell.UpdateVector( lastVector )

            # Check oCell for segment activation given active FCells.
            oCell.CheckSegmentActivation( self.activeFCells )

            # Perform learning on active segments in active oCell.
            oCell.SegmentLearning( self.activeFCells, self.winnerFCells, self.permanenceIncrement, self.permanenceDecrement,
                self.initialPermanence, self.maxSynapsesToAddPer, self.maxSynapsesPerSegment, self.FCellsPerColumn,
                self.FActivationThresholdMin, self.FActivationThresholdMax, self.initialPosVariance, self.segmentDecay, self.equalityThreshold )

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

        # Add active winner cells to working memory, with zero vector (since we're there), and no time steps, if it doesn't exist.
        self.workingMemory.append( [ self.winnerFCells, [ 0, 0 ], 0 ] )

    def PredictFCells( self, newVector ):
    # Clear old predicted FCells and generate new predicted FCells.

        predictedList = []
        for oCell in self.OCells:
            predictedList.append( oCell.GetPredicted( newVector, self.FCellsPerColumn * self.columnDimensions ) )

        self.predictedFCells = []
        for index, fCell in enumerate( self.FCells ):
            if fCell.CheckIfPredicted( index, predictedList, self.FActivationThresholdMin ):
                self.predictedFCells.append( index )

    def Compute( self, columnSDR, lastVector ):
    # Compute the action of vector memory, and learn on the synapses.

        self.columnSDR = columnSDR.sparse.tolist()

        # Safety check for column dimensions.
        if columnSDR.size != self.columnDimensions:
            print( "VM input column dimensions must be same as input SDR dimensions." )
            exit()

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells()

        self.ActivateOCells( lastVector )

        return None

from random import sample, randrange
from operator import add
from cell_and_synapse import FCell, OCell, WorkingMemory, BinarySearch, IndexIfItsIn, NoRepeatInsort, RepeatInsort, CheckInside
import numpy as np
from time import time

class VectorMemory:

    def __init__( self, columnDimensions, cellsPerColumn, numObjectCells, FActivationThresholdMin, FActivationThresholdMax, workingMemoryThreshold,
        initialPermanence, lowerThreshold, permanenceIncrement, permanenceDecrement, permanenceDecay, segmentDecay, initialPosVariance,
        ObjectRepActivaton, OActivationThreshold, maxSynapsesToAddPer, maxSegmentsPerCell, maxSynapsesPerSegment, equalityThreshold, pctAllowedOCellConns ):

        self.columnDimensions        = columnDimensions         # Dimensions of the column space.
        self.FCellsPerColumn         = cellsPerColumn           # Number of cells per column.
        self.numObjectCells          = numObjectCells           # Number of cells in the Object level.
        self.FActivationThresholdMin = FActivationThresholdMin  # Min threshold of active connected incident synapses...
        self.FActivationThresholdMax = FActivationThresholdMax  # Max threshold of active connected incident synapses...
                                                                # needed to activate segment.
        self.workingMemoryThreshold  = workingMemoryThreshold   # We have a different segment activation threshold for working memory.
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
        self.pctAllowedOCellConns    = pctAllowedOCellConns     # Percent of OCells an FCell can build connections to.

        # --------Create all the cells in the network.---------

        self.columnSDR    = []
        self.burstingCols = []

        # Create cells in feature layer.
        self.FCells = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell( initialPermanence, numObjectCells, pctAllowedOCellConns, segmentDecay ) )
        self.predictedFCells = []

        # Create cells in the object layer.
        self.OCells = []
        for i in range( numObjectCells ):
            self.OCells.append( OCell() )
        self.activeOCells = []

        self.lastActiveColumns = []

        self.workingMemory = WorkingMemory( 2 )

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

        return numActiveCells, int( sumSegs / numActiveCells ), self.workingMemory.stabilityScore

    def BuildLogData( self, log_data ):
    # Adds important information to log_data for entry into log.

        log_data.append( "Active Columns: " + str( len( self.columnSDR ) ) + ", " + str( self.columnSDR ) )

        activeFCells = []
        for index, fCell in enumerate( self.FCells ):
            if fCell.active:
                activeFCells.append( index )
        log_data.append( "Active F-Cells: " + str( len( activeFCells ) ) + ", " + str( activeFCells ) )

        log_data.append( "Bursting Column Pct: " + str( len( self.burstingCols ) / self.columnDimensions * 100 ) + "%" )
        log_data.append( "Bursting Columns: " + str( self.burstingCols ) )
        log_data.append( "Non-Bursting Columns: " + str( self.notBurstingCols ) )

        log_data.append( "Active O-Cells: " + str( len( self.activeOCells ) ) + ", " + str( self.activeOCells ) )

        log_data.append( "Predicted Cells: " + str( len( self.predictedFCells ) ) + ", " + str( self.predictedFCells ) )

        log_data.append( "Working Memory: " + str( self.workingMemory.ReturnSegmentData() ) )
        log_data.append( "Working Memory StabilityScore: " + str( self.workingMemory.stabilityScore ) )

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
        self.notBurstingCols = []

        for col in self.columnSDR:
            predictedCellsThisCol = []

            # Check if any cells in column are predicted. If yes then make them active.
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cell ].predicted:
                    predictedCellsThisCol.append( cell )

            if len( predictedCellsThisCol ) > 0:
                self.notBurstingCols.append( col )

                theCell = predictedCellsThisCol[ 0 ]
                for predCell in predictedCellsThisCol:
                    if self.FCells[ predCell ].ReturnActivationLevel() > self.FCells[ theCell ].ReturnActivationLevel():
                        theCell = predCell

                self.FCells[ theCell ].active = True
                self.FCells[ theCell ].winner = True

            # If none predicted then burst column, making all cells in column active.
            # Choose a winner cell at random.
            else:
                self.burstingCols.append( col )

                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    self.FCells[ cell ].active = True

                self.FCells[ randrange( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ) ].winner = True

        # If below threshold number of columns were predicted then burst every column.
        if len( self.burstingCols ) / self.columnDimensions >= 0.005:
            for col in self.notBurstingCols:
                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    self.FCells[ cell ].active = True

            self.burstingCols    = self.columnSDR.copy()
            self.notBurstingCols = []

    def ActivateOCells( self, workingMemorySegments, winnerCells ):
    # Use the segments of working memory to activate the OCells.

        # First check the stabilityScore of workingMemory, if it's below threshold then deactivate all OCells.
        if len( self.burstingCols ) / self.columnDimensions >= 0.005:
            for oCell in self.OCells:
                oCell.Deactivate()

        # If it's above threshold then check if there are OCells already active.
        else:
            # First convert the workingMemorySegments list into a bool list.
            workingMemorySegmentsBool = []
            for seg in workingMemorySegments:
                thisBool = []
                segIndex = 0
                for cell in range( self.FCellsPerColumn * self.columnDimensions ):
                    if segIndex < len( seg ) and seg[ segIndex ] == cell:
                        segIndex += 1
                        thisBool.append( True )
                    else:
                        thisBool.append( False )

                workingMemorySegmentsBool.append( thisBool )

            self.activeOCells = []
            for index, oCell in enumerate( self.OCells ):
                if oCell.active:
                    self.activeOCells.append( index )

            # If not then activate some by checking them against working memory.
            if len( self.activeOCells ) <= self.ObjectRepActivaton:
                for index, oCell in enumerate( self.OCells ):
                    if oCell.CheckOverlapAndActivate( workingMemorySegmentsBool ):
                        NoRepeatInsort( self.activeOCells, index )

            # If not enough activate in this way then activate random ones.
            while len( self.activeOCells ) <= self.ObjectRepActivaton:
                NoRepeatInsort( self.activeOCells, randrange( self.numObjectCells ) )

            # If too many OCells active then sort the list and remove the lowest overlap score ones.
            while len( self.activeOCells ) > self.ObjectRepActivaton:
                # Make a list of OCells total overlap score.
                allOCellOverlaps = []
                for aOCell in self.activeOCells:
                    allOCellOverlaps.append( self.OCells[ aOCell ].GetOverlapScore() )

                # Sort the overlaps and get the OCells with lowest overlap.
                oCellOverlapsSorted = [ i[ 0 ] for i in sorted( enumerate( allOCellOverlaps ), key = lambda x: x[ 1 ] ) ]

                del self.activeOCells[ oCellOverlapsSorted[ 0 ] ]

            # Perform learning on the active OCells.
            for aOCell in self.activeOCells:
                self.OCells[ aOCell ].OCellLearning( workingMemorySegmentsBool, winnerCells, self.FCellsPerColumn, self.permanenceIncrement, self.permanenceDecrement, self.initialPermanence, self.maxSynapsesToAddPer, self.maxSynapsesPerSegment, self.FActivationThresholdMin )

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

        self.predictedFCells = []

        # Get a bool list of all active and not active FCells.
        activeFCellsBool = []
        for fCell in self.FCells:
            activeFCellsBool.append( fCell.active )

        # Check every activeFCell's segments, and activate or deactivate them; make FCell predicted or not.
        for index, cell in enumerate( self.FCells ):

            # Make previously active segments lastActive (used in segment learning), and lastActive off.
            cell.UpdateSegmentActivity()

            # Check the cell against ative cells to see if any segment predicts this cell as terminal.
            if cell.CheckIfPredicted( activeFCellsBool, self.FCellsPerColumn, vector, self.lowerThreshold ):
                NoRepeatInsort( self.predictedFCells, index )

        # Check working memory for what cells it wants to predict.
        WMPredictedCells = self.workingMemory.GetPredictedCells( vector )
        for cell in WMPredictedCells:
            self.FCells[ cell ].predicted = True
            NoRepeatInsort( self.predictedFCells, cell )

    def Compute( self, columnSDR, lastVector ):
    # Compute the action of vector memory, and learn on the synapses.

        self.columnSDR = columnSDR.sparse.tolist()

        # Safety check for column dimensions.
        if columnSDR.size != self.columnDimensions:
            print( "VM input column dimensions must be same as input SDR dimensions." )
            exit()

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells()

        # Compile lists of last active FCells and lastActive winner FCells, for use below.
        activeCellsBool     = []
        winnerCells         = []
        lastActiveCellsBool = []
        lastWinnerCells     = []
        for index, fCell in enumerate( self.FCells ):
            activeCellsBool.append( fCell.active )
            lastActiveCellsBool.append( fCell.lastActive )
            if fCell.winner:
                winnerCells.append( index )
            if fCell.lastWinner:
                lastWinnerCells.append( index )

        # Update working memory vectors.
        self.workingMemory.Update( lastVector, activeCellsBool, winnerCells, self.permanenceIncrement, self.permanenceDecrement, self.initialPermanence, self.maxSegmentsPerCell, self.maxSynapsesPerSegment, self.segmentDecay, self.FCellsPerColumn, self.initialPosVariance, self.workingMemoryThreshold, self.lowerThreshold )

        # Use the segments in working memory to tell us what OCells to activate.
#        self.ActivateOCells( self.workingMemory.GetSegmentsAsList(), winnerCells )

        for cell in self.FCells:
            # Perform learning on Segments in FCells.
            cell.SegmentLearning( lastVector, lastActiveCellsBool, lastWinnerCells, self.initialPermanence, self.initialPosVariance,
                self.FActivationThresholdMin, self.FActivationThresholdMax, self.permanenceIncrement, self.permanenceDecrement,
                self.maxSynapsesToAddPer, self.maxSynapsesPerSegment, self.FCellsPerColumn, self.equalityThreshold )

#            self.OCellLearning()

        self.lastActiveColumns = self.columnSDR

        return None

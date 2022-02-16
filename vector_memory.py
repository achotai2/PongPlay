from random import sample, randrange
from operator import add
from cell_and_synapse import FCell, OCell, WorkingMemory, BinarySearch, IndexIfItsIn, NoRepeatInsort, RepeatInsort, CheckInside, FastIntersect
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

        self.columnSDR       = []
        self.burstingCols    = []
        self.notBurstingCols = []

        # Create cells in feature layer.
        self.FCells = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell( initialPermanence, numObjectCells, pctAllowedOCellConns, segmentDecay ) )
        self.activeFCells = []
        self.winnerFCells = []

        # Create cells in the object layer.
        self.OCells = []
        for i in range( numObjectCells ):
            self.OCells.append( OCell() )
        self.activeOCells = []

        self.workingMemory = WorkingMemory()

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
        log_data.append( "Winner F-Cells: " + str( self.winnerFCells ) )

        log_data.append( "Bursting Column Pct: " + str( len( self.burstingCols ) / self.columnDimensions * 100 ) + "%" )
        log_data.append( "Bursting Columns: " + str( self.burstingCols ) )
        log_data.append( "Non-Bursting Columns: " + str( self.notBurstingCols ) )

        log_data.append( "Active O-Cells: " + str( len( self.activeOCells ) ) + ", " + str( self.activeOCells ) )

        predictedFCells = []
        for index, fCell in enumerate( self.FCells ):
            if fCell.predicted:
                predictedFCells.append( index )
        log_data.append( "Predicted Cells: " + str( len( predictedFCells ) ) + ", " + str( predictedFCells ) )

        log_data.append( "Working Memory Entries: " + str( self.workingMemory ) )
        log_data.append( "Working Memory Stability Score: " + str( self.workingMemory.ReturnStabilityScore() ) )

    def ActivateFCells( self ):
    # Uses activated columns and cells in predicted state to put cells in active states.
    # Return a list of winnerCells.

        # Clean up old active and lastActive FCells.
        self.activeFCells = []
        self.winnerFCells = []
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
                    if self.FCells[ predCell ].HighestOverlapForActiveSegment() > self.FCells[ theCell ].HighestOverlapForActiveSegment():
                        theCell = predCell

                self.FCells[ theCell ].active = True
                self.FCells[ theCell ].winner = True
                NoRepeatInsort( self.activeFCells, theCell )
                NoRepeatInsort( self.winnerFCells, theCell )

            # If none predicted then burst column, making all cells in column active.
            # Choose a winner cell at random.
            else:
                self.burstingCols.append( col )

                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    self.FCells[ cell ].active = True
                    NoRepeatInsort( self.activeFCells, cell )

        # If the number of bursting columns is above threshold then burst every column.
#        if len( self.burstingCols ) / self.columnDimensions >= 0.005:
#            for col in self.notBurstingCols:
#                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
#                    self.FCells[ cell ].active = True
#                    NoRepeatInsort( self.activeFCells, cell )
#
#            self.notBurstingCols = []
#            self.burstingCols = self.columnSDR.copy()

        # Select the winner cells for all bursting columns.
        # First check the current active cells against working memory entry for overlap. If there's a high enough overlap then
        # we use the working memory as winner cells. If not then we choose winner cells randomly.
        inWorkingMem = self.workingMemory.SDROfEntry( [ 0, 0 ], self.initialPosVariance, self.activeFCells, self.FActivationThresholdMax )

        for col in self.burstingCols:
            theCell = -1
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if len( inWorkingMem ) > 0 and BinarySearch( inWorkingMem, cell ):
                    theCell = cell

            if theCell == -1:
                theCell = randrange( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn )

            self.FCells[ theCell ].winner = True
            NoRepeatInsort( self.winnerFCells, theCell )

    def ActivateOCells( self ):
    # First check the degree of bursting activity. If bursting above threshold then refresh all OCells. Then check against
    # FCell activity; we check OCells for activation of segments against activeFCells, even if bursting.

        # Check column bursting.
        if len( self.burstingCols ) / self.columnDimensions >= 0.005:
            self.activeOCells = []
            for oCell in self.OCells:
                oCell.Deactivate()

        # Build a bool version of working memory.
        activeFCellsBool = []
        winnerCells      = []
        for index, fCell in enumerate( self.FCells ):
            activeFCellsBool.append( fCell.active )
            if fCell.winner:
                winnerCells.append( index )

        # If OCells are active we feed only into the active ones. If not enough OCells are active we feed into all.
        checkOCells = []
        if len( self.activeOCells ) >= self.ObjectRepActivaton:
            checkOCells = self.activeOCells
        else:
            checkOCells = range( self.numObjectCells )

        # Feed FCell activation into the OCells to check for activation.
        for index, oCell in enumerate( checkOCells ):
            if self.OCells[ oCell ].CheckOverlapAndActivation( activeFCellsBool, 4, self.lowerThreshold ):
                NoRepeatInsort( self.activeOCells, index )
            else:
                actIndex = IndexIfItsIn( self.activeOCells, index )
                if actIndex != None:
                    del self.activeOCells[ actIndex ]

        # If the number of active OCells is below what we want then activate random ones.
        if len( self.activeOCells ) < self.ObjectRepActivaton:
            while len( self.activeOCells ) < self.ObjectRepActivaton:
                toActivate = randrange( self.numObjectCells )
                if not self.OCells[ toActivate ].active:
                    self.OCells[ toActivate ].active = True
                    NoRepeatInsort( self.activeOCells, toActivate )

        # If the number of OCells active is above the ObjectRepActivaton then check if the active OCells share a large
        # percentage of identical segments. If they do then randomly deactivate those that do.
# WHEN WE DO SEGMENT LEARNING WE MUST DECREASE SEGMENT SYNAPSES WHEN TERMINAL CELL IS INACTIVE.
# For now we'll just randomly check 10 oCells for equality, and delete ones at random.
#            oCellsToCheck = sample( range( self.numObjectCells ), 10 )
#            self.OCells[ randrange( self.numObjectCells ) ].Equality( self.OCells[ randrange ] )

# ACTUALLY, FOR NOW WE'LL JUST RANDOMLY DEACTIVE RANDOM ONES.
        while len( self.activeOCells ) > self.ObjectRepActivaton:
            toDeactivate = self.activeOCells.pop( randrange( len( self.activeOCells ) ) )
            self.OCells[ toDeactivate ].Deactivate()

        # Perform learning on the active OCells using the current active FCells.
        for aOCell in self.activeOCells:
            self.OCells[ aOCell ].OCellLearning( activeFCellsBool, winnerCells, self.FCellsPerColumn, self.lowerThreshold, self.permanenceIncrement,
                self.permanenceDecrement, self.initialPermanence, self.maxSynapsesToAddPer, self.maxSynapsesPerSegment, self.FActivationThresholdMin )

#    def OCellLearning( self ):
#    # Using the currently active and last active FCells, build stronger synapses to the active OCells,
#    # and weaken to the losers.
#
#        # Add all the active and last active OCells.
#        allActiveOCells = []
#        for actOCell in self.activeOCells:
#            NoRepeatInsort( allActiveOCells, actOCell )
#        for lActOCell in self.lastActiveOCells:
#            NoRepeatInsort( allActiveOCells, lActOCell )
#
#        # For all active and lastactive FCells strengthen to the active and lastActive OCells, and weaken to others.
#        for actOCell in self.activeOCells:
#            self.FCells[ actOCell ].OCellConnect( allActiveOCells, self.permanenceIncrement, self.permanenceDecrement )
#        for lActOCell in self.lastActiveOCells:
#            self.FCells[ lActOCell ].OCellConnect( allActiveOCells, self.permanenceIncrement, self.permanenceDecrement )

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

    def Compute( self, columnSDR, lastVector ):
    # Compute the action of vector memory, and learn on the synapses.

        self.columnSDR = columnSDR.sparse.tolist()

        # Safety check for column dimensions.
        if columnSDR.size != self.columnDimensions:
            print( "VM input column dimensions must be same as input SDR dimensions." )
            exit()

        # Update working memory entry at this location.
        self.workingMemory.UpdateVector( lastVector )

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells()

        # Compile lists of last active FCells and lastActive winner FCells, for use below.
        lastActiveCellsBool = []
        lastWinnerCells     = []
        for index, fCell in enumerate( self.FCells ):
            lastActiveCellsBool.append( fCell.lastActive )
            if fCell.lastWinner:
                lastWinnerCells.append( index )

        # Update working memory entry at this location.
        self.workingMemory.UpdateEntries( self.winnerFCells, self.segmentDecay, self.maxSynapsesPerSegment, self.initialPosVariance )

        # Use the active FCells to tell us what OCells to activate.
        self.ActivateOCells()

        for cell in self.FCells:
            # Perform learning on Segments in FCells.
            cell.SegmentLearning( lastVector, lastActiveCellsBool, lastWinnerCells, self.initialPermanence, self.initialPosVariance,
                self.FActivationThresholdMin, self.FActivationThresholdMax, self.permanenceIncrement, self.permanenceDecrement,
                self.maxSynapsesToAddPer, self.maxSynapsesPerSegment, self.FCellsPerColumn, self.equalityThreshold )

#            self.OCellLearning()

        return None

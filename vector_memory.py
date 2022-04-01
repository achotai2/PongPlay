from random import sample, randrange
from operator import add
from cell_and_synapse import FCell, OCell, SegmentStructure
from useful_functions import NoRepeatInsort, BinarySearch
#import numpy as np
#from time import time

class VectorMemory:

    def __init__( self, columnDimensions, cellsPerColumn, numObjectCells, FActivationThresholdMin, FActivationThresholdMax,
        initialPermanence, permanenceIncrement, permanenceDecrement, permanenceDecay, segmentDecay, objectRepActivation,
        maxSynapsesToAddPer, maxSynapsesPerSegment, equalityThreshold, vectorDimensions, initialVectorScaleFactor, initVectorConfidence, maxNonFeeling ):

        self.columnDimensions        = columnDimensions         # Dimensions of the column space.
        self.FCellsPerColumn         = cellsPerColumn           # Number of cells per column.
        self.numObjectCells          = numObjectCells           # Number of cells in the Object level.
        self.FActivationThresholdMin = FActivationThresholdMin  # Min threshold of active connected incident synapses...
        self.FActivationThresholdMax = FActivationThresholdMax  # Max threshold of active connected incident synapses...
                                                                # needed to activate segment.
        self.initialPermanence       = initialPermanence        # Initial permanence of a new synapse.
        self.permanenceIncrement     = permanenceIncrement      # Amount by which permanences of synapses are incremented during learning.
        self.permanenceDecrement     = permanenceDecrement      # Amount by which permanences of synapses are decremented during learning.
        self.permanenceDecay         = permanenceDecay          # Amount to decay permances each time step if < 1.0.
        self.segmentDecay            = segmentDecay             # If a segment hasn't been active in this many time steps then delete it.
        self.objectRepActivation     = objectRepActivation      # Number of active OCells in object layer at one time.
        self.maxSynapsesToAddPer     = maxSynapsesToAddPer      # The maximum number of FToFSynapses added to a segment during creation.
        self.maxSynapsesPerSegment   = maxSynapsesPerSegment    # Maximum number of active synapses allowed on a segment.
        self.equalityThreshold       = equalityThreshold        # The number of equal synapses for two segments to be considered identical.
        self.vectorDimensions        = vectorDimensions         # The number of dimensions of our vector space.
        self.initVectorScaleFactor   = initialVectorScaleFactor
        self.initVectorConfidence    = initVectorConfidence
        self.maxNonFeeling           = maxNonFeeling

        # Create column SDR storage.
        self.columnSDR       = []
        self.burstingCols    = []
        self.notBurstingCols = []

        # Create cells in feature layer.
        self.FCells = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell() )
        self.activeFCells     = []
        self.lastActiveFCells = []
        self.winnerFCells     = []
        self.lastWinnerFCells = []
        self.predictedFCells  = []

         # Stores and deals with all FCell to FCell segments.
        self.FToFSegmentStruct = SegmentStructure( vectorDimensions, initialPermanence, permanenceIncrement, permanenceDecrement, permanenceDecay,
            FActivationThresholdMin, FActivationThresholdMax, columnDimensions, cellsPerColumn, maxSynapsesToAddPer, maxSynapsesPerSegment,
            segmentDecay, equalityThreshold, initialVectorScaleFactor, initVectorConfidence, maxNonFeeling  )

        # Create cells in object layer.
        self.OCells = []
        for o in range( numObjectCells ):
            self.OCells.append( OCell() )
        self.activeOCells = []

        self.stateOCellData = []            # Stores the data for the active O-Cells Report.

    def SendData( self, stateNumber ):
    # Return the active FCells as a list.

        for fCell in self.FCells:
            fCell.ReceiveStateData( stateNumber )

#        if len( self.stateOCellData ) > 0 and self.stateOCellData[ -1 ][ 0 ] == None:
#            self.stateOCellData[ -1 ][ 0 ] = stateColour

    def GetStateInformation( self ):
    # Get all the cells state information and return it.

        fCellsToReturn = []
        for fCell in self.FCells:
            fCellsToReturn.append( fCell.ReturnStateInformation() )

        return fCellsToReturn, self.stateOCellData

    def GetGraphData( self ):
    # Return the number of active cells.

        return len( self.activeFCells ), self.FToFSegmentStruct.HowManyActiveSegs(), len( self.predictedFCells )

    def BuildLogData( self, log_data ):
    # Adds important information to log_data for entry into log.

        log_data.append( "Active Columns: " + str( len( self.columnSDR ) ) + ", " + str( self.columnSDR ) )

        log_data.append( "Active F-Cells: " + str( len( self.activeFCells ) ) + ", " + str( self.activeFCells ) )
        log_data.append( "Winner F-Cells: " + str( self.winnerFCells ) )

        log_data.append( "Active O-Cells: " + str( len( self.activeOCells ) ) + ", " + str( self.activeOCells ) )

        log_data.append( "Bursting Column Pct: " + str( len( self.burstingCols ) / self.columnDimensions * 100 ) + "%" )
        log_data.append( "Bursting Columns: " + str( self.burstingCols ) )
        log_data.append( "Non-Bursting Columns: " + str( self.notBurstingCols ) )

        log_data.append( "Predicted Cells: " + str( len( self.predictedFCells ) ) + ", " + str( self.predictedFCells ) )

#        log_data.append( "Working Memory Entries: " + str( self.workingMemory ) )
#        log_data.append( "Working Memory Stable: " + str( self.workingMemory.reachedStability ) )

        log_data.append( "# of FToF-Segments: " + str( len( self.FToFSegmentStruct.segments ) ) + ", # of Active Segments: " + str( len( self.FToFSegmentStruct.activeSegments ) ) )

    def ActivateFCells( self ):
    # Uses activated columns and cells in predicted state to put cells in active states.
    # Return a list of winnerCells.

        # Clean up old active, lastActive, winner, and lastWinner FCells.
        for lActCell in self.lastActiveFCells:
            self.FCells[ lActCell ].lastActive = False
        self.lastActiveFCells = self.activeFCells.copy()
        for actCell in self.activeFCells:
            self.FCells[ actCell ].active     = False
            self.FCells[ actCell ].lastActive = True
        self.activeFCells = []

        for lWinCell in self.lastWinnerFCells:
            self.FCells[ lWinCell ].lastWinner = False
        self.lastWinnerFCells = self.winnerFCells.copy()
        for winCell in self.winnerFCells:
            self.FCells[ winCell ].winner     = False
            self.FCells[ winCell ].lastWinner = True
        self.winnerFCells = []

        self.burstingCols    = []
        self.notBurstingCols = []

        for col in self.columnSDR:
            predictedCellsThisCol = []
            activeThisColumn = []
            winnerThisColumn = -1

            # Get working memories suggestion for this column.
            wmCells = self.FToFSegmentStruct.GetWinnerCellForColumn( col )

            # Check if any cells in column are predicted. If yes then make a note of them.
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cell ].predicted:
                    predictedCellsThisCol.append( cell )

            # If result was predicted by FCells or working memory...
            if len( predictedCellsThisCol ) > 0:
                self.notBurstingCols.append( col )

                for winCell in wmCells:
                    if BinarySearch( predictedCellsThisCol, winCell ):
                        activeThisColumn.append( winCell )
                        break
                winnerThisColumn = activeThisColumn[ -1 ]

            # If result wasn't predicted by FCells...
            else:
                self.burstingCols.append( col )

                # Make all cells in column active.
                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    activeThisColumn.append( cell )

                # If working memory has a prediction for this column then make it the winner. It'll be random anyways.
                winnerThisColumn = wmCells[ 0 ]

            # Make the chosen cells active and winners.
            if len( activeThisColumn ) > 0:
                for toAct in activeThisColumn:
                    self.FCells[ toAct ].active = True
                    self.activeFCells.append( toAct )
            else:
                print( "No cells chosen active this column.")
                exit()
            if winnerThisColumn != -1:
                self.FCells[ winnerThisColumn ].winner = True
                self.winnerFCells.append( winnerThisColumn )
            else:
                print( "No cells chosen winner this column.")
                exit()

    def PredictFCells( self, vector ):
    # Clear old predicted FCells and generate new predicted FCells.

        # Refresh old prediction.
        for cell in self.FCells:
            cell.predicted = False

        if vector[ 2 ] == 0:
            # Get the predicted FCells and make them predicted state.
            self.predictedFCells = self.FToFSegmentStruct.GetStimulatedSegments( self.activeFCells, vector )

            # Make the selected cells predicted state.
            for predCell in self.predictedFCells:
                self.FCells[ predCell ].predicted = True

    def Compute( self, columnSDR, lastVector ):
    # Compute the action of vector memory, and learn on the synapses.

        if lastVector[ 2 ] != 0:
            self.FToFSegmentStruct.ResetWorkingMemory()

        self.columnSDR = columnSDR.sparse.tolist()

        # Safety check for column dimensions.
        if columnSDR.size != self.columnDimensions:
            print( "VM input column dimensions must be same as input SDR dimensions." )
            exit()

        # Update working memory.
        self.FToFSegmentStruct.UpdateWorkingMemory( lastVector )

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells()

        # If any winner cell was not predicted then create a new segment to the lastActive FCells.
        for winCell in self.winnerFCells:
            if not self.FCells[ winCell ].predicted:
                self.FToFSegmentStruct.CreateSegment( self.FCells, self.lastWinnerFCells, winCell, lastVector )

        # Perform learning on segments.
        self.FToFSegmentStruct.SegmentLearning( self.FCells, self.lastWinnerFCells, self.lastActiveFCells, True )

        return None

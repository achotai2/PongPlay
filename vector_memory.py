from random import sample, randrange
from operator import add
from cell_and_synapse import FCell, OCell, WorkingMemory, SegmentStructure, BinarySearch, IndexIfItsIn, NoRepeatInsort, RepeatInsort, CheckInside, FastIntersect
#import numpy as np
#from time import time

class VectorMemory:

    def __init__( self, columnDimensions, cellsPerColumn, numObjectCells, FActivationThresholdMin, FActivationThresholdMax,
        initialPermanence, lowerThreshold, permanenceIncrement, permanenceDecrement, permanenceDecay, segmentDecay,
        initialPosVariance, objectRepActivation, OActivationThreshold, maxSynapsesToAddPer, maxSegmentsPerCell,
        maxSynapsesPerSegment, equalityThreshold, pctAllowedOCellConns , WMStabilityThreshold, vectorDimensions,
        numVectorSynapses, vectorRange, vectorScaleFactor ):

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
        self.objectRepActivation     = objectRepActivation       # Number of active OCells in object layer at one time.
        self.OActivationThreshold    = OActivationThreshold     # Threshold of active connected OToFSynapses...
                                                                # needed to activate OCell.
        self.maxSynapsesToAddPer     = maxSynapsesToAddPer       # The maximum number of FToFSynapses added to a segment during creation.
        self.maxSegmentsPerCell      = maxSegmentsPerCell       # The maximum number of segments per cell.
        self.maxSynapsesPerSegment   = maxSynapsesPerSegment     # Maximum number of active synapses allowed on a segment.
        self.equalityThreshold       = equalityThreshold        # The number of equal synapses for two segments to be considered identical.
        self.pctAllowedOCellConns    = pctAllowedOCellConns     # Percent of OCells an FCell can build connections to.
        self.WMyStabilityThreshold   = WMStabilityThreshold     # The threshold of stable segments in working memory for it to be considered stable
        self.vectorDimensions        = vectorDimensions         # The number of dimensions of our vector space.
        self.numVectorSynapses       = numVectorSynapses        # The number of vector synapses in segments.
        self.vectorRange             = vectorRange              # The initial total range of vector views in segments.
        self.vectorScaleFactor       = vectorScaleFactor        # The adjustment of vector permanences in segments off-center.

        # Create column SDR storage.
        self.columnSDR       = []
        self.burstingCols    = []
        self.notBurstingCols = []

        # Create cells in feature layer.
        self.FCells = []
        for i in range( columnDimensions * cellsPerColumn ):
            self.FCells.append( FCell( initialPermanence, numObjectCells, pctAllowedOCellConns, segmentDecay ) )
        self.activeFCells     = []
        self.lastActiveFCells = []
        self.winnerFCells     = []
        self.lastWinnerFCells = []
        self.predictedFCells  = []

         # Stores and deals with all FCell to FCell segments.
        self.FToFSegmentStruct = SegmentStructure( vectorDimensions, initialPermanence, permanenceIncrement, permanenceDecrement,
            permanenceDecay, FActivationThresholdMin, FActivationThresholdMax, columnDimensions, cellsPerColumn,
            maxSynapsesToAddPer, maxSynapsesPerSegment, segmentDecay, equalityThreshold, numVectorSynapses, vectorRange, vectorScaleFactor )

        # Create cells in object layer.
        self.OCells = []
        for o in range( numObjectCells ):
            self.OCells.append( OCell() )
        self.activeOCells = []

        self.stateOCellData = []            # Stores the data for the active O-Cells Report.

        # Stores and deals with all the OCell to FCell (Working Memory) segments.
        self.OToFSegmentStruct = SegmentStructure( 0, initialPermanence, permanenceIncrement, permanenceDecrement,
            permanenceDecay, FActivationThresholdMin, FActivationThresholdMax, columnDimensions, cellsPerColumn,
            maxSynapsesToAddPer, maxSynapsesPerSegment, segmentDecay, equalityThreshold, numVectorSynapses, vectorRange, vectorScaleFactor )
        self.OToOSegmentStruct = SegmentStructure( 0, initialPermanence, permanenceIncrement, permanenceDecrement,
            permanenceDecay, FActivationThresholdMin, FActivationThresholdMax, numObjectCells, 1,
            maxSynapsesToAddPer, maxSynapsesPerSegment, segmentDecay, equalityThreshold, numVectorSynapses, vectorRange, vectorScaleFactor )


        self.workingMemory = WorkingMemory( FActivationThresholdMax, initialPosVariance, cellsPerColumn, segmentDecay, WMStabilityThreshold, vectorDimensions )

    def SendData( self, stateNumber, stateColour ):
    # Return the active FCells as a list.

        for fCell in self.FCells:
            fCell.ReceiveStateData( stateNumber )

        if len( self.stateOCellData ) > 0 and self.stateOCellData[ -1 ][ 0 ] == None:
            self.stateOCellData[ -1 ][ 0 ] = stateColour

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

        log_data.append( "Working Memory Entries: " + str( self.workingMemory ) )
        log_data.append( "Working Memory Stable: " + str( self.workingMemory.reachedStability ) )

#        log_data.append( "# of Segments: " + str( len( self.FToFSegmentStruct.segments ) ) + ", # of Active Segments: " + str( len( self.FToFSegmentStruct.activeSegments ) ) )

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
            wmCell = self.workingMemory.GetCellForColumn( col )

            # Check if any cells in column are predicted. If yes then make a note of them.
            for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                if self.FCells[ cell ].predicted:
                    predictedCellsThisCol.append( cell )

            # If result was predicted by FCells or working memory...
            if len( predictedCellsThisCol ) > 0 or ( self.workingMemory.reachedStability and wmCell != None ):
                self.notBurstingCols.append( col )

                # If working memory is stable then it selects the winner.
                if self.workingMemory.reachedStability and wmCell != None:
                    activeThisColumn.append( wmCell )
                    winnerThisColumn = wmCell
                # Otherwise select from predictedCellsThisCol the cell with most activation.
                else:
                    activeThisColumn.append( self.FToFSegmentStruct.ThereCanBeOnlyOne( predictedCellsThisCol )[ 0 ] )
                    winnerThisColumn = activeThisColumn[ -1 ]

            # If result wasn't predicted by FCells...
            else:
                self.burstingCols.append( col )

                # Make all cells in column active.
                for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
                    activeThisColumn.append( cell )

                # If working memory has a prediction for this column then make it the winner.
                if wmCell != None:
                    winnerThisColumn = wmCell
                # If not then select a random cell in column as winner.
                else:
                    winnerThisColumn = randrange( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn )

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

    def ActivateOCells( self ):
    # If working memory switches from unstable to stable then check all OCells against working memory,
    # activate them and perform learning on them.

        # Get the list of stable segments from working memory.
        workingMemorySegments = self.workingMemory.ReturnSegmentsAsList()

        # Feed working memory segments into OCells and get their activation.
        for entry in workingMemorySegments:
            oCellsForSegment = self.OToFSegmentStruct.GetStimulatedCells( entry, [] )
            for actOCell in oCellsForSegment:
                self.OCells[ actOCell ].AddActivation( self.OToFSegmentStruct.ThereCanBeOnlyOne( [ actOCell ] )[ 1 ] )

        # Use activation to check which, if any, OCells become active.
        aboveSegmentThreshold = []                                   # Index of OCells above segment threshold.
        activationLevel       = []                                   # The degree of total segement activation for each OCell.
        for index, oCell in enumerate( self.OCells ):
            if oCell.CheckSegmentActivationLevel( 4 ):
                aboveSegmentThreshold.append( index )
                activationLevel.append( oCell.ReturnOverallActivation() )
        oCellActivationSortedIndices = sorted( range( len( activationLevel ) ), key = lambda k: activationLevel[ k ] )

        toActIndex = 0
        while len( self.activeOCells ) < self.objectRepActivation:
            if toActIndex < len( oCellActivationSortedIndices ):
                toAct = aboveSegmentThreshold[ oCellActivationSortedIndices[ len( oCellActivationSortedIndices ) - toActIndex - 1 ] ]
                toActIndex += 1
            else:
                toAct = randrange( self.numObjectCells )
                # If we're activating random OCells then create new segments on them.
                for entry in workingMemorySegments:
                    self.OToFSegmentStruct.CreateSegment( self.OCells, entry, toAct, [], None )

            self.OCells[ toAct ].active = True
            NoRepeatInsort( self.activeOCells, toAct )

# I PROBABLY DO WANT TO DO DECAY AND CREATE ON THESE SEGMENTS EVENTUALLY, I JUST NEED TO FIGURE OUT HOW (MAYBE SEND A LIST?).
        self.OToFSegmentStruct.SegmentLearning( self.OCells, [], False )

        # Enter the data into the final report data list.
        self.stateOCellData.append( [ None, self.activeOCells ] )

    def RefreshOCells( self ):
    # If working memory switched from stable to unstable then deactivate all OCells.

        self.activeOCells = []

        for oCell in self.OCells:
            oCell.ResetState()

    def PredictFCells( self, vector ):
    # Clear old predicted FCells and generate new predicted FCells.

        # Refresh old prediction.
        for cell in self.FCells:
            cell.predicted = False

        # Get the predicted FCells and make them predicted state.
        self.predictedFCells = self.FToFSegmentStruct.GetStimulatedCells( self.activeFCells, vector )
        for predCell in self.predictedFCells:
            self.FCells[ predCell ].predicted = True

    def Compute( self, columnSDR, lastVector ):
    # Compute the action of vector memory, and learn on the synapses.

        self.columnSDR = columnSDR.sparse.tolist()

        # Safety check for column dimensions.
        if columnSDR.size != self.columnDimensions:
            print( "VM input column dimensions must be same as input SDR dimensions." )
            exit()

        # Update working memory entry at this location.
        stableBefore = self.workingMemory.reachedStability
        self.workingMemory.UpdateVectorAndReceiveColumns( lastVector, self.columnSDR )
        stableAfter = self.workingMemory.reachedStability

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells()

        # Update working memory entry at this location.
        self.workingMemory.UpdateEntries( self.columnSDR, self.winnerFCells )

        # If any winner cell was not predicted then create a new segment to the lastActive FCells.
        for winCell in self.winnerFCells:
            if not self.FCells[ winCell ].predicted:
                self.FToFSegmentStruct.CreateSegment( self.FCells, self.lastWinnerFCells, winCell, lastVector, None )

        # Perform learning on segments.
        self.FToFSegmentStruct.SegmentLearning( self.FCells, self.lastWinnerFCells, True )

        # If working memory stability changed then modify OCell states.
        if stableBefore and not stableAfter:
            self.RefreshOCells()
        elif not stableBefore and stableAfter:
            self.ActivateOCells()

        return None

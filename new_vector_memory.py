from random import sample, randrange
from operator import add
from segment_struct import SegmentStructure
from cell_struct import FCell
from useful_functions import NoRepeatInsort, BinarySearch, ReturnMaxIndices, FastIntersect
#import numpy as np
from time import time

class NewVectorMemory:

    def __init__( self, vectorMemoryDict ):

        self.vectorMemoryDict = vectorMemoryDict

        # Create column SDR storage.
        self.columnSDR       = []
        self.lastColumnSDR   = []
        self.burstingCols    = []
        self.notBurstingCols = []

        # Create cells in feature layer.
        self.FCells = []
        for i in range( vectorMemoryDict[ "columnDimensions" ] * vectorMemoryDict[ "cellsPerColumn" ] ):
            self.FCells.append( FCell( int( i / vectorMemoryDict[ "cellsPerColumn" ] ), vectorMemoryDict ) )
        self.activeFCells     = []
        self.lastActiveFCells = []
        self.winnerFCells     = []
        self.lastWinnerFCells = []
        self.predictedFCells  = []

         # Stores and deals with all FCell to FCell segments.
        self.FToFSegmentStruct = SegmentStructure( vectorMemoryDict )

        # Vector portion.
        self.newVectorSDR  = []
        self.lastVectorSDR = []

    def SendData( self, stateNumber ):
    # Return the active FCells as a list.

        for fCell in self.FCells:
            fCell.ReceiveStateData( stateNumber )

    def GetStateInformation( self ):
    # Get all the cells state information and return it.

        fCellsToReturn = []
        for fCell in self.FCells:
            fCellsToReturn.append( fCell.ReturnStateInformation() )

        return fCellsToReturn

    def GetGraphData( self ):
    # Return the number of active cells.

        return len( self.activeFCells ), self.FToFSegmentStruct.HowManyActiveSegs(), len( self.predictedFCells )

    def BuildLogData( self, log_data ):
    # Adds important information to log_data for entry into log.

        log_data.append( "Active Columns: " + str( len( self.columnSDR ) ) + ", " + str( self.columnSDR ) )

        log_data.append( "Active F-Cells: " + str( len( self.activeFCells ) ) + ", " + str( self.activeFCells ) )
        log_data.append( "Winner F-Cells: " + str( self.winnerFCells ) )

#        log_data.append( "Active O-Cells: " + str( len( self.activeOCells ) ) + ", " + str( self.activeOCells ) )

        log_data.append( "Bursting Column Pct: " + str( len( self.burstingCols ) / self.vectorMemoryDict[ "numActiveColumnsPerInhArea" ] * 100 ) + "%" )
        log_data.append( "Bursting Columns: " + str( self.burstingCols ) )
        log_data.append( "Non-Bursting Columns: " + str( self.notBurstingCols ) )

        log_data.append( "Predicted Cells: " + str( len( self.predictedFCells ) ) + ", " + str( self.predictedFCells ) )

        log_data.append( "# of FToF-Segments: " + str( self.FToFSegmentStruct.HowManySegs() )
            + ", # of Active Segments: " + str( self.FToFSegmentStruct.HowManyActiveSegs() )
            + ", # of Winner Segments: " + str( self.FToFSegmentStruct.HowManyWinnerSegs() )
            )

    def ThereCanBeOnlyOne( self, activeCellsList ):
    # Choose the winner cell for column by choosing active one with highest terminal activation.

        # Find the predicted cell
        greatestActivation = 0
        greatestCell       = activeCellsList[ 0 ]

        if len( activeCellsList ) > 1:
            for cell in activeCellsList:
                if self.FCells[ cell ].GetTerminalActivation() > greatestActivation:
                    greatestActivation = self.FCells[ cell ].GetTerminalActivation()
                    greatestCell       = cell

        return greatestCell

    def ActivateFCells( self ):
    # Uses activated columns and cells in predicted state to put cells in active states.
    # Return a list of winnerCells.

        # Move active cells to last active, and lastactive to not active.
        self.UpdateFCellActivity()

        self.burstingCols    = []
        self.notBurstingCols = []

        for col in self.columnSDR:
            predictedCellsThisCol = []
            activeThisColumn = []
            winnerThisColumn = -1

            # Check if any cells in column are predicted. If yes then make a note of them.
            for cell in range( col * self.vectorMemoryDict[ "cellsPerColumn" ], ( col * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
                if self.FCells[ cell ].predicted:
                    predictedCellsThisCol.append( cell )

            # If result was predicted by FCells...
            if len( predictedCellsThisCol ) > 0:
                self.notBurstingCols.append( col )

                # Choose the winner cell.
                winnerThisColumn = self.ThereCanBeOnlyOne( predictedCellsThisCol )
                # Make only winner active.
                activeThisColumn.append( winnerThisColumn )

            # If result wasn't predicted by FCells...
            else:
                self.burstingCols.append( col )

                # Make all cells in column active.
                for cell in range( col * self.vectorMemoryDict[ "cellsPerColumn" ], ( col * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
                    activeThisColumn.append( cell )

                # If working memory has a prediction for this column then make it the winner.
                winnerThisColumn = randrange( col * self.vectorMemoryDict[ "cellsPerColumn" ], ( col * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] )

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

    def PredictFCells( self ):
    # Clear old predicted FCells and generate new predicted FCells.

        # Refresh old prediction.
        for cell in self.FCells:
            cell.predicted = False
        self.predictedFCells = []

        # Get the predicted FCells and make them predicted state.
        self.predictedFCells = self.FToFSegmentStruct.StimulateSegments( self.FCells, self.activeFCells, self.newVectorSDR )

        # Make the selected cells predicted state.
        for predCell in self.predictedFCells:
            self.FCells[ predCell ].predicted = True

    def UpdateFCellActivity( self ):
    # Updates the FCell states for a next time step.

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

    def GetMotorVector( self ):
    # Check if the winning predicted segments have above threshold confidence. If they do then return a motor vector of appropriate size.

        if self.FToFSegmentStruct.SegmentsAreConfident():
            return self.FToFSegmentStruct.GetVector()

    def Compute( self, columnSDR, newVectorSDR ):
    # Compute the action of vector memory, and learn on the synapses.

        self.lastColumnSDR = self.columnSDR.copy()
        self.columnSDR     = columnSDR.sparse.tolist()

        self.lastVectorSDR = self.newVectorSDR.copy()
        self.newVectorSDR  = newVectorSDR.copy()

        # Safety check for column dimensions.
        if columnSDR.size != self.vectorMemoryDict[ "columnDimensions" ]:
            print( "VM input column dimensions must be same as input SDR dimensions." )
            exit()

        # Clear old active cells and get new ones active cells for this time step.
        startTime = time()
        self.ActivateFCells()
        print( "Activate: " + str( time() - startTime ) )

        # If any winner cell was not predicted (bursting) then create a new segment terminal to it.
        startTime = time()
        if len( self.burstingCols ) / self.vectorMemoryDict[ "numActiveColumnsPerInhArea" ] > 0.5:
            self.FToFSegmentStruct.CreateSegment( self.FCells, self.lastColumnSDR, self.columnSDR, self.lastWinnerFCells, self.winnerFCells, self.lastVectorSDR )

#        for burCol in self.burstingCols:
#            for burCell in range( burCol * self.vectorMemoryDict[ "cellsPerColumn" ], ( burCol * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
#                if self.FCells[ burCell ].winner:
#                    self.FToFSegmentStruct.CreateSegment( self.FCells, self.lastColumnSDR, burCol, self.lastWinnerFCells, burCell, self.lastVectorSDR )
#                    break
        print( "Create Segment: " + str( time() - startTime ) )

        # Perform learning on segments.
        startTime = time()
        self.FToFSegmentStruct.SegmentLearning( self.FCells, self.activeFCells, self.lastActiveFCells )
        print( "Segment Learning: " + str( time() - startTime ) )

        # Refresh segment states.
        startTime = time()
        self.FToFSegmentStruct.UpdateSegmentActivity( self.FCells )
        print( "Update Actvitiy: " + str( time() - startTime ) )

# COULD EVEN SPEED THIS UP BY NOT RUNNING. SIMPLY WHEN WE GO TO ACTIVE CELLS WE USE THE ACTIVE TERMINAL COLUMNS TO GET THE SEGMENTS WHICH ARE TERMINAL THERE AND CHECK THEM ONLY.
        startTime = time()
        self.PredictFCells()
        print( "Predict: " + str( time() - startTime ) )

        return self.GetMotorVector()

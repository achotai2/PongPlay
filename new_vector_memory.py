from random import sample, randrange
from operator import add
from segment_struct import SegmentStructure
from cell_struct import FCell
from useful_functions import NoRepeatInsort, BinarySearch, ReturnMaxIndices, FastIntersect
from numpy import setdiff1d
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

    def ChooseWinnerSegmentAndCells( self, lastVectorSDR, lastPositionSDR ):
    # Use the active segments from last time step, plus the plus the last motor vectorSDR, plus the presently active columns, to choose a winner segment(s).
    # Also choose winnerFCells.

        if self.FToFSegmentStruct.ChooseWinnerSegment( self.columnSDR, lastVectorSDR, lastPositionSDR ):
            # Get the winnercells from the winner segment.
            winnerSegmentTerminalCells, winnerSegmentTerminalCols = self.FToFSegmentStruct.ReturnWinnerCells()

            # Get the bursting and not bursting columns from the returned winner cells from winner segment.
            self.notBurstingCols = FastIntersect( self.columnSDR, winnerSegmentTerminalCols )
            self.burstingCols    = setdiff1d( self.columnSDR, self.notBurstingCols, True )

            # For all the returned cells we make them winner.
            if len( self.notBurstingCols ) > 0:
                for i, col in enumerate( winnerSegmentTerminalCols ):
                    if BinarySearch( self.notBurstingCols, col ):
                        NoRepeatInsort( self.winnerFCells, winnerSegmentTerminalCells[ i ] )

            # If there are active columns with no returned winner then make random cell active.
            if len( self.burstingCols ) > 0:
                for col in self.burstingCols:
                    NoRepeatInsort( self.winnerFCells, randrange( col * self.vectorMemoryDict[ "cellsPerColumn" ], ( col * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ) )

        else:
            # Randomly choose winner cells for each column.
            for col in self.columnSDR:
                NoRepeatInsort( self.winnerFCells, randrange( col * self.vectorMemoryDict[ "cellsPerColumn" ], ( col * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ) )

            # If no valid winner segment then create a new segment by randomly selecting winner terminal cells and make this segment the winner.
            self.FToFSegmentStruct.CreateSegment( self.FCells, self.lastColumnSDR, self.columnSDR, self.lastWinnerFCells, self.winnerFCells, lastVectorSDR, lastPositionSDR )

            # Make all columns bursting.
            self.burstingCols    = self.columnSDR.copy()
            self.notBurstingCols = []

        # Make the chosen winner cells winner.
        for winCell in self.winnerFCells:
            self.FCells[ winCell ].SetAsWinner()

        if len( self.winnerFCells ) > 40:
            print( "ChooseWinnerSegmentAndCells(): Function generated too many winner cells (>40)." )
            exit()

    def ActivateFCells( self ):
    # Uses activated columns and cells in predicted state to put cells in active states.
    # Return a list of winnerCells.

        # Make the not-bursting columns winner cells active.
        for col in self.notBurstingCols:
            for cell in range( col * self.vectorMemoryDict[ "cellsPerColumn" ], ( col * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
                if self.FCells[ cell ].IsWinner():
                    self.FCells[ cell ].MakeActive()
                    NoRepeatInsort( self.activeFCells, cell )

        # Make every cell in the bursting columns active.
        for col in self.burstingCols:
            for cell in range( col * self.vectorMemoryDict[ "cellsPerColumn" ], ( col * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
                self.FCells[ cell ].MakeActive()
                NoRepeatInsort( self.activeFCells, cell )

    def PredictFCells( self ):
    # Clear old predicted FCells and generate new predicted FCells.

# CHANGE THIS TO BE RUN IF WE WANT PREDICTED CELLS GIVEN A PARTICULAR VECTOR AND INCIDENT CELLS, SO IT RECEIVES THE INCIDENT CELLS AND VECTOR AND RETURNS PREDICTION.

        # Stimulate the segments using the incident cells.
        self.FToFSegmentStruct.StimulateSegments( self.FCells, self.activeFCells )

        # Use the stimulated segments to get the most confident segments to return a vector and predictedCells.
        self.predictedFCells, motorVector = self.FToFSegmentStruct.ChooseVectorSegment( self.FCells )

        # Make the selected cells predicted state.
        for predCell in self.predictedFCells:
            self.FCells[ predCell ].predicted = True

        return motorVector

    def ActivateSegments( self, lastVectorSDR, lastPositionSDR ):
    # Perform learning on segments and refresh segments, then activate new ones.

        self.FToFSegmentStruct.SegmentLearning( self.FCells, self.activeFCells, self.lastActiveFCells, lastVectorSDR, lastPositionSDR )

        # Refresh segment states.
        self.FToFSegmentStruct.UpdateSegmentActivity( self.FCells )

        # Activate segments using the current active FCells as incident cells.
        self.FToFSegmentStruct.StimulateSegments( self.FCells, self.activeFCells )

    def GetMotorVectorSDR( self ):
    # Find if there is a confident segment, and if so then get its motor vector SDR.

        return self.FToFSegmentStruct.ChooseVectorSegment() [ 1 ]

    def UpdateFCellActivity( self ):
    # Updates the FCell states for a next time step.

# I SHOULD PROBABLY CALL A FUNCTION IN THE CELLS TO DO THIS INSTEAD.
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

        # Refresh old prediction.
        for pCell in self.predictedFCells:
            self.FCells[ pCell ].predicted = False
        self.predictedFCells = []

        # Refresh the column states.
        self.burstingCols    = []
        self.notBurstingCols = []

    def Compute( self, columnSDR, lastVectorSDR, lastPositionSDR ):
    # Compute the action of vector memory, and learn on the synapses.

        self.lastColumnSDR = self.columnSDR.copy()
        self.columnSDR     = columnSDR.sparse.tolist()

        # Safety check for column dimensions.
        if columnSDR.size != self.vectorMemoryDict[ "columnDimensions" ]:
            print( "VM input column dimensions must be same as input SDR dimensions." )
            exit()

        # Move active cells to last active, and lastactive to not active, and predicted to not.
        self.UpdateFCellActivity()

        # Choose a winner segment(s) from all active segments from last time step. Also chosoe winner FCells.
        self.ChooseWinnerSegmentAndCells( lastVectorSDR, lastPositionSDR )

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells()

        # Perform learning on segments and refresh segments, then activate new ones.
        self.ActivateSegments( lastVectorSDR, lastPositionSDR )

        # Get motor vector from active segments.
        newMotorSDR = self.GetMotorVectorSDR()

        return newMotorSDR

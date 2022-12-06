from random import sample, randrange
from operator import add
from cell_and_synapse import FCell, OCell, SegmentStructure
from working_memory import WorkingMemory
from useful_functions import NoRepeatInsort, BinarySearch, ReturnMaxIndices, FastIntersect
#import numpy as np
#from time import time

class VectorMemory:

    def __init__( self, vectorMemoryDict ):

        self.vectorMemoryDict = vectorMemoryDict

        # Create column SDR storage.
        self.columnSDR       = []
        self.burstingCols    = []
        self.notBurstingCols = []

        # Create cells in feature layer.
        self.FCells = []
        for i in range( vectorMemoryDict[ "columnDimensions" ] * vectorMemoryDict[ "cellsPerColumn" ] ):
            self.FCells.append( FCell() )
        self.activeFCells     = []
        self.lastActiveFCells = []
        self.winnerFCells     = []
        self.lastWinnerFCells = []
        self.predictedFCells  = []

         # Stores and deals with all FCell to FCell segments.
        self.FToFSegmentStruct = SegmentStructure( vectorMemoryDict )

#        self.workingMemory = WorkingMemory( vectorMemoryDict )

        # Create cells in object layer.
#        self.OCells = []
#        for o in range( vectorMemoryDict[ "numObjectCells" ] ):
#            self.OCells.append( OCell() )
#        self.activeOCells = []
#        self.OCellFeeling = []

#        self.stateReflectTime = 0
#        self.lastOCellSDR = None

        self.stateOCellData = []            # Stores the data for the active O-Cells Report.

    def SendData( self, stateNumber ):
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

#        log_data.append( "Active O-Cells: " + str( len( self.activeOCells ) ) + ", " + str( self.activeOCells ) )

        log_data.append( "Bursting Column Pct: " + str( len( self.burstingCols ) / self.vectorMemoryDict[ "columnDimensions" ] * 100 ) + "%" )
        log_data.append( "Bursting Columns: " + str( self.burstingCols ) )
        log_data.append( "Non-Bursting Columns: " + str( self.notBurstingCols ) )

        log_data.append( "Predicted Cells: " + str( len( self.predictedFCells ) ) + ", " + str( self.predictedFCells ) )

#        log_data.append( "Working Memory Entries: " + str( self.workingMemory ) )
#        log_data.append( "Working Memory Stable: " + str( self.workingMemory.reachedStability ) )

        log_data.append( "# of FToF-Segments: " + str( len( self.FToFSegmentStruct.segments ) ) + ", # of Active Segments: " + str( len( self.FToFSegmentStruct.activeSegments ) ) +
            ", # of Stimulated Segments: " + str( len( self.FToFSegmentStruct.stimulatedSegments ) ) )

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

#            # Get working memories suggestion for this column.
#            wmCell = self.workingMemory.GetCellForColumn( col )

            # Check if any cells in column are predicted. If yes then make a note of them.
            for cell in range( col * self.vectorMemoryDict[ "cellsPerColumn" ], ( col * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
                if self.FCells[ cell ].predicted:
                    predictedCellsThisCol.append( cell )

            # If result was predicted by FCells...
#            if len( predictedCellsThisCol ) > 0 or ( self.workingMemory.reachedStability and wmCell != None ):
            if len( predictedCellsThisCol ) > 0:
              self.notBurstingCols.append( col )

#                # If working memory is stable then it selects the winner.
#                if self.workingMemory.reachedStability and wmCell != None:
#                    activeThisColumn.append( wmCell )
#                    winnerThisColumn = wmCell
#                # Otherwise select from predictedCellsThisCol the cell with most activation.
#                else:

                # Make all predicted cells active.
                activeThisColumn = predictedCellsThisCol.copy()

                # Choose the winner cell.
                winnerThisColumn = self.FToFSegmentStruct.ThereCanBeOnlyOne( predictedCellsThisCol, col )

            # If result wasn't predicted by FCells...
            else:
                self.burstingCols.append( col )

                # Make all cells in column active.
                for cell in range( col * self.vectorMemoryDict[ "cellsPerColumn" ], ( col * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
                    activeThisColumn.append( cell )

                # If working memory has a prediction for this column then make it the winner.
#                if wmCell != None:
#                    winnerThisColumn = wmCell
                # If not then select a random cell in column as winner.
#                else:
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

#    def ActivateOCells( self ):
#    # Use the FToFSegmentStruct stimulated and active segments to determine the active OCells.
#
#        # Refresh old active OCells.
#        for actOCell in self.activeOCells:
#            self.OCells[ actOCell ].active = False
#        self.activeOCells = []
#
#        # Gather the stimulation level for all OCells from the stimulated FToFSegmentStruct segments.
#        stimulatedOCellCounts = self.FToFSegmentStruct.GetStimulatedOCells( self.vectorMemoryDict[ "numObjectCells" ] )
#
#        # Get the OCells with the highest counts and make these active.
#        stimulatedOCells = ReturnMaxIndices( stimulatedOCellCounts, self.vectorMemoryDict[ "objectRepActivation" ], True )
#        for oCell in stimulatedOCells:
#            if stimulatedOCellCounts[ oCell ] >= self.vectorMemoryDict[ "OCellActivationThreshold" ]:
#                self.activeOCells.append( oCell )
#
#        for actOCell in self.activeOCells:
#            self.OCells[ actOCell ].active = True

#    def CheckOCellFeeling( self ):
#    # Check the active OCell rep against stored OCell feeling states. If one exists then return this feeling.
#
#        maxOverlap = 0
#        maxFeeling = 0.0
#
#        for entry in self.OCellFeeling:
#            overlap = len( FastIntersect( self.activeOCells, entry[ 0 ] ) )
#            if overlap >= self.vectorMemoryDict[ "objectRepActivation" ] and overlap > maxOverlap:
#                maxFeeling = entry[ 1 ]
#
#        if maxFeeling > 0.0:
#            print( "POSITIVE Feeling" )
#        elif maxFeeling < 0.0:
#            print( "NEGATIVE Feeling" )
#
#        return maxFeeling

    def PredictFCells( self, vector ):
    # Clear old predicted FCells and generate new predicted FCells.

        # Refresh old prediction.
        for cell in self.FCells:
            cell.predicted = False

        # Get the predicted FCells and make them predicted state.
        self.predictedFCells = self.FToFSegmentStruct.StimulateSegments( self.activeFCells, vector )

        # Make the selected cells predicted state.
        for predCell in self.predictedFCells:
            self.FCells[ predCell ].predicted = True

#    def Memorize( self, feeling ):
#    # Save the present state in working memory for unique learning later, during reflection.
#    # Then clear working memory entries and FToFSegmentStruct stimulated segments.
#
#        self.workingMemory.SaveState( feeling )

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

#    def Refresh( self ):
#    # Clear the entries of working memory, all stimulated segmnets, and all predicted and active FCells, and all active OCells.
#
#        for pCell in self.predictedFCells:
#            self.FCells[ pCell ].predicted = False
#        self.predictedFCells = []
#
#        for aCell in self.activeFCells:
#            self.FCells[ aCell ].active = False
#        self.activeFCells = []
#
#        for lCell in self.lastActiveFCells:
#            self.FCells[ lCell ].lastActive = False
#        self.lastActiveFCells = []
#
#        for lwCell in self.lastWinnerFCells:
#            self.FCells[ lwCell ].lastWinner = False
#        self.lastWinnerFCells = []
#
#        for wCell in self.winnerFCells:
#            self.FCells[ wCell ].winner = False
#        self.winnerFCells = []
#
#        for oCell in self.activeOCells:
#            self.OCells[ oCell ].active = False
#        self.activeOCells = []
#
#        self.workingMemory.Reset()
#
#        self.FToFSegmentStruct.ResetStimulatedSegments()
#        self.workingMemory.Reset()

#    def DeleteSavedState( self ):
#    # Order workingMemory to delete its Zeroth saved state. Check if working memory has any more states left and return this.
#
#        self.workingMemory.DeleteSavedStateEntry()
#
#        return self.workingMemory.StillReflecting()

#    def ChooseNextReflectionEntry( self ):
#    # Randomly choose and return the next workingMemory entry index.
#
#        return self.workingMemory.ReturnRandomEntryIndex()

#    def GenerateUniqueCellReps( self, stateFeeling ):
#    # Tell workingMemory to generate random cells for each columnSDR stored.
#    # Also generate a random OCell activation rep.
#
#        self.activeOCells = []
#
#        # Check if there is an OCell rep stored for this feelingState.
#        for entry in self.OCellFeeling:
#            if entry[ 1 ] == stateFeeling:
#                self.activeOCells = entry[ 0 ]
#        if len( self.activeOCells ) == 0:
#            self.activeOCells = sorted( sample( range( self.vectorMemoryDict[ "numObjectCells" ] ), self.vectorMemoryDict[ "objectRepActivation" ] ) )
#            # Save the new OCell rep with feeling.
#            self.OCellFeeling.append( [ self.activeOCells.copy(), stateFeeling ] )

#        # Generate unique FCell reps for each stored column, randomly if they don't exist stored.
#        savedActiveOCells = self.workingMemory.GenerateCells( self.lastOCellSDR )

        # Generate a unique OCell rep.
#        if len( savedActiveOCells ) == 0:
#            self.activeOCells = sorted( sample( range( self.vectorMemoryDict[ "numObjectCells" ] ), self.vectorMemoryDict[ "objectRepActivation" ] ) )
#            self.lastOCellSDR = None
#        else:
#            self.activeOCells = savedActiveOCells
#            self.lastOCellSDR = savedActiveOCells

#        for oCell in self.activeOCells:
#            self.OCells[ oCell ].active = True

#    def VectorMemoryReflect( self, lastWMEntryID, thisWMEntryID, nextWMEntryID ):
#    # Go through a reflective period to learn important stored states with a unique OCell represention and more specific vector and SDR thresholds.
#
#        print( "Vector Memory Reflecting..." )
#
#        # Calculate the vectors given last and this WMEntryIDs.
#        lastVector = self.workingMemory.CalculateVector( lastWMEntryID, thisWMEntryID )
#        nextVector = self.workingMemory.CalculateVector( thisWMEntryID, nextWMEntryID )
#
#        # Update FCell activity.
#        self.UpdateFCellActivity()
#
#        # Get this columnSDR.
#        self.columnSDR = self.workingMemory.GetEntrySDR( thisWMEntryID )
#
#        # Activate the FCells.
#        self.ActivateFCells()
#
#        # Update working memory.
#        self.workingMemory.UpdateVectorAndReceiveColumns( lastVector, self.columnSDR, self.activeFCells, self.activeOCells )
#        # Update entries of working memory with active cells.
#        self.workingMemory.UpdateEntries( self.winnerFCells )
#
#        # If any winner cell was not predicted then create a new segment to the lastActive FCells.
#        for winCell in self.winnerFCells:
#            if not self.FCells[ winCell ].predicted:
#                self.FToFSegmentStruct.CreateSegment( self.FCells, self.lastWinnerFCells, winCell, lastVector, self.activeOCells )
#
#        # Perform learning on segments.
#        self.FToFSegmentStruct.SegmentLearning( self.FCells, self.OCells, self.lastWinnerFCells, self.lastActiveFCells, self.activeOCells, lastVector )
#
#        # Refresh segment states.
#        self.FToFSegmentStruct.UpdateSegmentActivity( self.FCells )
#
#        # Predict the next FCells.
#        self.PredictFCells( nextVector )

    def Compute( self, columnSDR, lastVector ):
    # Compute the action of vector memory, and learn on the synapses.

        print( "Vector Memory Computing..." )

        self.columnSDR = columnSDR.sparse.tolist()

        # Safety check for column dimensions.
        if columnSDR.size != self.vectorMemoryDict[ "columnDimensions" ]:
            print( "VM input column dimensions must be same as input SDR dimensions." )
            exit()

        # Clear old active cells and get new ones active cells for this time step.
        self.ActivateFCells()

#        # Gather the segment averages, and send them to working memory to update its entry averages.
#        self.workingMemory.UpdateAverages( self.FToFSegmentStruct.GetSegmentAverages( self.FCells ) )

#        # Stimulate and activate OCells using the stimulated and active segments.
#        self.ActivateOCells()
#        self.CheckOCellFeeling()

#        # Update working memory.
#        self.workingMemory.UpdateVectorAndReceiveColumns( lastVector, self.columnSDR, self.activeFCells, self.activeOCells )
#        # Update entries of working memory with active cells.
#        self.workingMemory.UpdateEntries( self.winnerFCells )

#        # If any winner cell was not predicted then create a new segment to the lastActive FCells.
        for winCell in self.winnerFCells:
            if not self.FCells[ winCell ].predicted:
                self.FToFSegmentStruct.CreateSegment( self.FCells, self.lastWinnerFCells, winCell, lastVector )

        # Perform learning on segments.
        self.FToFSegmentStruct.SegmentLearning( self.FCells, self.lastWinnerFCells, self.lastActiveFCells, lastVector )

        # Refresh segment states.
        self.FToFSegmentStruct.UpdateSegmentActivity( self.FCells )

from segment_struct import SegmentStructure
from cell_struct import Cell
from useful_functions import IndexOfGreatest

class SpatialPooler:

    def __init__( self, vectorMemoryDict, inputDimensions ):
    # Initialize new SpatialPooler.

        self.vectorMemoryDict = vectorMemoryDict

        self.inputDimensions  = inputDimensions

        # Create columns for this system.
        self.Columns = []
        for i in range( vectorMemoryDict[ "columnDimensions" ] ):
            self.Columns.append( Cell( int( i ), vectorMemoryDict ) )

         # Stores and deals with all input to output column segments.
        self.IToOSegmentStruct = SegmentStructure( vectorMemoryDict )

    def CreateRepresentation( self, newInput ):
    # Create new segments using most likely representation and boosting.

        # Get the cells of the top 10 most active segments, if any exist.
        terminalColumnsList = self.IToOSegmentStruct.ReturnCellsFromStimulatedSegs( 10 )

        terminalColumns = []
        occuranceCounts = []
        for item in terminalColumnsList:
            for col in item:
                numCols = len( terminalColumns )
                insertIndex = NoRepeatInsort( terminalColumns, col )
                if len( terminalColumns ) == numCols:
                    occuranceCounts[ insertIndex ] += 1
                else:
                    occuranceCounts.insert( insertIndex, 0 )


    def Compute( self, inputList ):
    # Take the input active cells and compute active columns.

        # Refresh segment states.
        self.FToFSegmentStruct.UpdateSegmentActivity( self.FCells )

        # Use the inputList active cells to stimulate segments.
        self.IToOSegmentStruct.StimulateSegments( self.Columns, inputList )

        terminalColumns = self.IToOSegmentStruct.ReturnHighestOverlapActiveSegment()

        if len( terminalColumns ) == 0:
            self.CreateRepresentation( inputList )
        else:
            return terminalColumns

from useful_functions import BinarySearch, NoRepeatInsort, CheckInside, GenerateUnitySDR, FastIntersect

class WorkingMemory:

    def __init__( self, FCellsPerColumn, FColumnDimensions, vectorDimensions, numVectorSynapses ):
    # Setup working memory.

        self.FCellsPerColumn    = FCellsPerColumn
        self.FColumnDimensions  = FColumnDimensions
        self.vectorDimensions   = vectorDimensions
        self.numVectorSynapses  = numVectorSynapses

        self.positionCellScores = [ [] for c in range( FColumnDimensions ) ]

        self.currentPosition    = []
        for d in range( vectorDimensions ):
            self.currentPosition.append( 0 )

    def __repr__( self ):
    # Returns properties of this class as a string.

        stringReturn = ""

        for entryIdx in range( len( self.entryCellSDR ) ):
            stringReturn = ( stringReturn + "\n Entry #" +
                str( entryIdx ) + " - Pos: " + str( self.entryPos[ entryIdx ] ) +
                " - Time: " + str( self.entryTime[ entryIdx ] ) +
                " - ColumnSDR: " + str( self.entryColumnSDR[ entryIdx ] ) +
                " - # of entryCellSDR: " + str( len( self.entryCellSDR[ entryIdx ] ) ) +
                " - SDR: " + str( self.entryCellSDR[ entryIdx ] ) )

        return stringReturn

    def UpdatePositionCellScores( self, newPositionCellScores ):
    # Given this time steps position-cell score data, update self.positionCellScores.

        for cell in newPositionCellScores:
            columnEntry = self.positionCellScores[ int( cell[ 0 ] / self.FCellsPerColumn ) ]

            # Using the pre-existing cell entries for this column update the average score; or create a new entry.
            cellIndex = 0
            while cellIndex < len( columnEntry ):
                if columnEntry[ cellIndex ][ 0 ] == cell[ 0 ]:
                    break
                else:
                    cellIndex += 1

            if cellIndex == len( columnEntry ):
                # Calculate weighted average.
                newScores = [ [] for i in range( self.vectorDimensions ) ]
                for d in range( self.vectorDimensions ):
                    for x in range( self.numVectorSynapses ):
                         newScores[ d ].append( cell[ 2 ][ d ][ x ] / cell[ 1 ] )

                columnEntry.append( [ cell[ 0 ], cell[ 1 ], newScores ] )        # [ Terminal Cell, Count, Position Scores ]
            else:
                newScores = [ [] for i in range( self.vectorDimensions ) ]
                for d in range( self.vectorDimensions ):
                    for x in range( self.numVectorSynapses ):
                         newScores[ d ].append( ( ( columnEntry[ cellIndex ][ 1 ] * columnEntry[ cellIndex ][ 2 ][ d ][ x ] ) + cell[ 2 ][ d ][ x ] ) / ( columnEntry[ cellIndex ][ 1 ] + cell[ 1 ] ) )

                columnEntry[ cellIndex ][ 1 ] += cell[ 1 ]
                columnEntry[ cellIndex ][ 2 ] = newScores

    def UpdateVector( self, vector ):
    # Use the vector to update the local workingMemory position.

        if len( vector ) != self.vectorDimensions:
            print( "Vectors sent to working memory of wrong size." )
            exit()

        # Update vector.
        for d in range( self.vectorDimensions ):
            self.currentPosition[ d ] += vector[ d ]

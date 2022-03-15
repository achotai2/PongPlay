from useful_functions import BinarySearch, NoRepeatInsort, CheckInside, GenerateUnitySDR, FastIntersect
from cell_and_synapse import SegmentStructure

class WorkingMemory:

    def __init__( self, vectorRange, maxTimeSinceActive, sameThreshold, vectorDimensions, initialPermanence,
        permanenceIncrement, permanenceDecrement, permanenceDecay, activationThreshold, columnDimensions,
        cellsPerColumn, maxSynapsesToAddPer, maxSynapsesPerSegment, equalityThreshold, numVectorSynapses, vectorScaleFactor ):
    # Setup working memory.

        self.WMToFSegments   = SegmentStructure( vectorDimensions, initialPermanence, permanenceIncrement, permanenceDecrement,
            permanenceDecay, activationThreshold, activationThreshold, columnDimensions, cellsPerColumn, maxSynapsesToAddPer,
            maxSynapsesPerSegment, maxTimeSinceActive, equalityThreshold, numVectorSynapses, vectorRange, vectorScaleFactor )

        self.currentLocation = []
        for x in range( vectorDimensions ):
            self.currentLocation.append( 0 )

        self.thisEntryIndex   = None
        self.reachedStability = False               # True if all entries history fits above sameThreshold.
        self.timeSame         = 0

        self.sameThreshold    = sameThreshold
        self.vectorDimensions = vectorDimensions

    def __repr__( self ):
    # Returns properties of this class as a string.

        stringReturn = ""

#        for entryIdx in range( len( self.entryCellSDR ) ):
#            stringReturn = ( stringReturn + "\n Entry #" +
#                str( entryIdx ) + " - Pos: " + str( self.entryPos[ entryIdx ] ) +
#                " - Time: " + str( self.entryTime[ entryIdx ] ) +
#                " - ColumnSDR: " + str( self.entryColumnSDR[ entryIdx ] ) +
#                " - # of entryCellSDR: " + str( len( self.entryCellSDR[ entryIdx ] ) ) +
#                " - SDR: " + str( self.entryCellSDR[ entryIdx ] ) )

        return stringReturn

    def UpdateVectorAndReceiveColumns( self, vector, columnSDR ):
    # Use the vector to update the present location.
    # Then, given the columnSDR, check overlap and vector for segments that agree above threshold.
    # If none agrees then create one at this location, use the vectorCells for
    # If one exists then it becomes activated and we perform learning on it using winnerCellSDR.
    # Also check if working memory has reached stability, meaning that it hasn't had to create a new segment in a while.

        # Update vector.
        if len( vector ) != self.vectorDimensions:
            print( "Vectors sent to working memory of wrong size." )
            exit()

        for x in range( self.vectorDimensions ):
            self.currentLocation[ x ] += vector[ x ]

        # Check columnSDR and vector and activate valid segments.
        self.thisEntryIndex = self.WMToFSegments.CheckSegmentColumnOverlap( columnSDR, self.currentLocation )

        # Check if working memory becomes stable.
        if self.thisEntryIndex != None:
            self.timeSame += 1
        else:
            self.timeSame = 0
        if self.timeSame >= self.sameThreshold:
            self.reachedStability = True
        else:
            self.reachedStability = False

    def UpdateEntries( self, winnerCells ):
    # Perform learning on working memory segments.

        if self.thisEntryIndex != None:
            # Perform learning on segments.
            self.WMToFSegments.SegmentLearning( [], winnerCells, True )
        else:
            self.WMToFSegments.CreateSegment( [], winnerCells, None, self.currentLocation, [] )

    def GetCellForColumn( self, col ):
    # For the predicted entry and given column return the cell working memory predicted, if it does.
    # If it didn't predict this column return None.

        if self.thisEntryIndex != None:
            return self.WMToFSegments.CellForColumn( self.thisEntryIndex, col )

        return None

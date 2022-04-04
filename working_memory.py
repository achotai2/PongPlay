from useful_functions import BinarySearch, NoRepeatInsort, CheckInside, GenerateUnitySDR, FastIntersect

class WorkingMemory:

    def __init__( self, overlapThreshold, initialPosVariance, FCellsPerColumn, maxTimeSinceActive, sameThreshold,
        samePercent, vectorDimensions ):
    # Setup working memory.

        self.entryColumnSDR  = []                   # The column input for each entry.
        self.entryCellSDR    = []                   # The last 3 cell inputs for each entry.
        self.unityCellSDR    = []                   # The most seen cell entry for last three. If tie choose least winner.
        self.entryPos        = []                   # The current relative position for each entry.
        self.entryTime       = []                   # The time since active for entry.

        self.thisEntryIndex   = None                  # The zero vector location entry index.
        self.columnSDRFits    = False               # True if the zero vector entry fits the column input.
        self.reachedStability = False               # True if all entries history fits above sameThreshold.

        self.overlapThreshold   = overlapThreshold
        self.checkRange         = initialPosVariance
        self.sameThreshold      = sameThreshold
        self.samePercent        = samePercent
        self.FCellsPerColumn    = FCellsPerColumn
        self.maxTimeSinceActive = maxTimeSinceActive
        self.vectorDimensions   = vectorDimensions

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

    def Reset( self ):
    # Refresh all entries of working memory.

        self.entryColumnSDR  = []
        self.entryCellSDR    = []
        self.unityCellSDR    = []
        self.entryPos        = []
        self.entryTime       = []

        self.thisEntryIndex   = None
        self.columnSDRFits    = False
        self.reachedStability = False    

    def UpdateVectorAndReceiveColumns( self, vector, columnSDR ):
    # Use the vector to update all vectors stored in workingMemory items, and add timeStep.
    # Then, given the columnSDR, check overlap at zero-location. If above threshold make note of index.
    # Also check if reached stability, if it has then calculate unityCellSDR.

        # Update vector.
        if len( self.entryCellSDR ) > 0:
            if len( vector ) != len( self.entryPos[ 0 ] ):
                print( "Vectors sent to working memory of wrong size." )
                exit()

            for index in range( len( self.entryCellSDR ) ):
                for x in range( len( self.entryPos[ index ] ) ):
                    self.entryPos[ index ][ x ] += vector[ x ]

        # Check columnSDR
        self.thisEntryIndex = None
        for entryIdx, entry in enumerate( self.entryColumnSDR ):
            if CheckInside( [ 0 for i in range( self.vectorDimensions ) ], self.entryPos[ entryIdx ], self.checkRange ):
                self.thisEntryIndex = entryIdx

                if len( FastIntersect( columnSDR, entry ) ) >= self.overlapThreshold:
                    self.columnSDRFits = True
                else:
                    self.columnSDRFits = False

        # If zero entry doesn't exist then create a new one.
        if self.thisEntryIndex == None:
            self.thisEntryIndex = len( self.entryColumnSDR )
            self.entryColumnSDR.append( columnSDR )
            self.entryCellSDR.append( [] )
            self.entryPos.append( [ 0 for i in range( self.vectorDimensions ) ] )
            self.entryTime.append( 0 )
            self.unityCellSDR.append( [] )

        # If the entry doesn't fit then clear the zero-vector cell-SDRs and modify the columnSDR.
        if not self.columnSDRFits:
            self.entryCellSDR[ self.thisEntryIndex ]   = []
            self.entryColumnSDR[ self.thisEntryIndex ] = columnSDR

        # Check if reached stability.
        self.reachedStability = True
        if not self.columnSDRFits:
            self.reachedStability = False
        else:
            numStable = 0
            for entry in self.entryCellSDR:
                if len( entry ) >= self.sameThreshold:
                    numStable += 1
            if numStable / len( self.entryColumnSDR ) < self.samePercent:
                self.reachedStability = False

        # Update unityCellSDR.
        self.unityCellSDR = []
        for entry in self.entryCellSDR:
            if len( entry ) > 0:
                self.unityCellSDR.append( GenerateUnitySDR( entry, len( columnSDR ), self.FCellsPerColumn ) )
            else:
                self.unityCellSDR.append( [] )

    def UpdateEntries( self, columnSDR, winnerCells ):
    # If any item in working memory is above threshold time steps then remove it.
    # Add the new active Fcells to working memory at the 0-vector location, append it on the end, if columnSDRFits.

        entryToDelete = []

        # Insert winner cells into entry SDR.
        if self.thisEntryIndex != None:
            self.entryCellSDR[ self.thisEntryIndex ].append( winnerCells )

            if self.columnSDRFits:
                self.entryTime[ self.thisEntryIndex ] = -1
            for entryIdx in range( len( self.entryCellSDR ) ):
                # Update time step.
                self.entryTime[ entryIdx ] += 1

                if self.entryTime[ entryIdx ] > self.maxTimeSinceActive:
                    NoRepeatInsort( entryToDelete, entryIdx )

        # For any entry with entryCellSDR greater than threshold delete the oldest ones.
        for entry in self.entryCellSDR:
            while len( entry ) > self.sameThreshold:
                del entry[ 0 ]

        # Delete items whose timeStep is above threshold.
        if len( entryToDelete ) > 0:
            for toDel in reversed( entryToDelete ):
                del self.entryColumnSDR[ toDel ]
                del self.entryCellSDR[ toDel ]
                del self.unityCellSDR[ toDel ]
                del self.entryPos[ toDel ]
                del self.entryTime[ toDel ]

    def GetCellForColumn( self, col ):
    # For the predicted entry and given column return the cell working memory predicted, if it does.
    # If it didn't predict this column return None.

        for cell in range( col * self.FCellsPerColumn, ( col * self.FCellsPerColumn ) + self.FCellsPerColumn ):
            if self.thisEntryIndex != None:
                if BinarySearch( self.unityCellSDR[ self.thisEntryIndex ], cell ):
                    return cell
        return None

    def ReturnSegmentsAsList( self ):
    # Return the unityCellSDR cells as a list of lists.

        toReturn = []

        for entry in self.unityCellSDR:
            toReturn.append( entry )

        return toReturn

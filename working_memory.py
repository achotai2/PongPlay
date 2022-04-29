from random import randrange
from useful_functions import BinarySearch, NoRepeatInsort, CheckInside, GenerateUnitySDR, FastIntersect, CalculateDistanceScore

class WorkingMemory:

    def __init__( self, vectorMemoryDict ):
    # Setup working memory.

        self.vectorMemoryDict = vectorMemoryDict

        self.currentPosition = []
        for d in range( vectorMemoryDict[ "vectorDimensions" ] ):
            self.currentPosition.append( 0 )

        self.entryColumnSDR  = []                   # The column input for each entry.
        self.entryFCellSDR   = []                   # The last 3 cell inputs for each entry.
        self.unityFCellSDR   = []                   # The last 3 cell inputs for each entry.
#        self.entryOCellSDR   = []
        self.entryCenter     = []                   # The current relative origin position for each entry.
        self.entryTime       = []                   # The time since active for entry.
        self.entryCount      = []                   # The number of FToF segments which contributed to this entry.
        self.standardDeviation  = []                   # The average distance scalar for this entry.
        self.vectConfidence  = []                   # The average vector confidence for this entry.
        self.SDRThreshold    = []                   # The average SDR overlap threshold for this entry.

        self.thisEntryIndex   = None                  # The zero vector location entry index.
        self.columnSDRFits    = False               # True if the zero vector entry fits the column input.
        self.reachedStability = False               # True if all entries history fits above sameThreshold.

        self.savedStates     = []
#        self.thisStateFCells = []

        self.lastEntryID = 0
        self.thisEntryID = 0

    def __repr__( self ):
    # Returns properties of this class as a string.

        stringReturn = ""

        for entryIdx in range( len( self.entryColumnSDR ) ):
            stringReturn += "\n"
            if entryIdx == self.thisEntryIndex:
                stringReturn += "*"
            stringReturn += ( "Entry #" + str( entryIdx ) +
                " - Pos Center: " + str( self.entryCenter[ entryIdx ] ) +
                " - Time: " + str( self.entryTime[ entryIdx ] ) +
                " - Count: " + str( self.entryCount[ entryIdx ] ) +
                " - vectorConfidence: " + str( self.vectConfidence[ entryIdx ] ) +
                " - standardDeviation: " + str( self.standardDeviation[ entryIdx ] ) +
                " - ColumnSDR: " + str( self.entryColumnSDR[ entryIdx ] ) +
                " - # of entryCellSDR: " + str( len( self.entryFCellSDR[ entryIdx ] ) ) +
                " - SDR: " + str( self.entryFCellSDR[ entryIdx ] ) )

        return stringReturn

    def SaveState( self, feeling ):
    # Save the present column and center state of working memory for use during reflection.

        if len( self.savedStates ) == 0:
            self.savedStates.append( [ self.entryColumnSDR.copy(), self.entryCenter.copy() ] )

    def DeleteSavedStateEntry( self ):
    # Delete the Zeroth savedStates entry as we are done reflecting on it.

        print( len( self.savedStates ) )

        if len( self.savedStates ) > 0:
            del self.savedStates[ 0 ]

#    def ReturnStateFeeling( self ):
#    # Return the saved feeling of the zeroth state.
#
#        return self.savedStates[ 0 ][ 2 ]

    def ReturnRandomEntryIndex( self ):
    # Return a random index for the zeroth entry of savedStates.

        return randrange( len( self.savedStates[ 0 ][ 0 ] ) )

    def CalculateVector( self, lastEntryID, thisEntryID ):
    # Calculate and return the vector from lastEntryID to thisEntryID.

        vector = []
        for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
            vector.append( self.savedStates[ 0 ][ 1 ][ thisEntryID ][ d ] - self.savedStates[ 0 ][ 1 ][ lastEntryID ][ d ] )

        return vector

    def GenerateRandomCells( self, columnSDR ):
    # For the zeroth saved state entry generate and store a random cell rep for each columnSDR.

        entryCellSDR = []

        for col in columnSDR:
            entryCellSDR.append( randrange( 0, self.vectorMemoryDict[ "cellsPerColumn" ] ) + ( col * self.vectorMemoryDict[ "cellsPerColumn" ] ) )

        return entryCellSDR

#    def GenerateCells( self, lastOCellSDR ):
#    # For the zero entry check if the entry already has cells stored. If yes then fine, if not then generate random entries.
#
#        self.thisStateFCells = []
#        OCellSDR             = []
#
#        for entryIdx in range( len( self.savedStates[ 0 ][ 0 ] ) ):
#            if len( self.savedStates[ 0 ][ 3 ][ entryIdx ] ) == len( self.savedStates[ 0 ][ 0 ][ entryIdx ] ):
#                self.thisStateFCells.append( self.savedStates[ 0 ][ 3 ][ entryIdx ].copy() )
#            else:
#                self.thisStateFCells.append( self.GenerateRandomCells( self.savedStates[ 0 ][ 0 ][ entryIdx ] ) )
#
#            OCellSDR = self.savedStates[ 0 ][ 4 ]
#
#        if lastOCellSDR == None or OCellSDR == []:
#            return OCellSDR
#        else:
#            return GenerateUnitySDR( [ OCellSDR, lastOCellSDR ], self.vectorMemoryDict[ "objectRepActivation" ], 1 )

    def GetEntrySDR( self, thisEntryID ):
    # Get and return thisEntryID columnSDR.

        return self.savedStates[ 0 ][ 0 ][ thisEntryID ]

    def StillReflecting( self ):
    # Return False if the savedStates list is empty, otherwise return True.

        if len( self.savedStates ) > 0:
            return True
        else:
            return False

    def Reset( self ):
    # Refresh all entries of working memory.

        self.currentPosition = []
        for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
            self.currentPosition.append( 0 )

        self.entryColumnSDR  = []
        self.entryFCellSDR   = []
        self.unityFCellSDR   = []
#        self.entryOCellSDR   = []
        self.entryCenter     = []
        self.entryTime       = []
        self.entryCount      = []
        self.standardDeviation  = []
        self.vectConfidence  = []
        self.SDRThreshold    = []

        self.thisEntryIndex   = None
        self.columnSDRFits    = False
        self.reachedStability = False

        self.thisEntryID = 0
        self.lastEntryID = 0

    def UpdateVectorAndReceiveColumns( self, vector, columnSDR, FCellSDR, OCellSDR ):
    # Use the vector to update all vectors stored in workingMemory items, and add timeStep.
    # Then, given the columnSDR, check overlap at zero-location. If above threshold make note of index.
    # Also check if reached stability, if it has then calculate unityCellSDR.

        # Update vector.
        if len( vector ) != self.vectorMemoryDict[ "vectorDimensions" ]:
            print( "Vectors sent to working memory of wrong size." )
            exit()

        for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
            self.currentPosition[ d ] += vector[ d ]

        # Check columnSDR against all entries, and their vector against vectorScoreThreshold.
        self.thisEntryIndex = None
        self.columnSDRFits  = False
        fittingEntries      = []
        fittingScores       = []
        for entryIdx, entry in enumerate( self.entryColumnSDR ):
            vectorScore  = CalculateDistanceScore( self.currentPosition, self.vectorMemoryDict[ "vectorDimensions" ], self.entryCenter[ entryIdx ], self.standardDeviation[ entryIdx ], self.vectConfidence[ entryIdx ] )
            overlapScore = len( FastIntersect( columnSDR, entry ) )

            if vectorScore >= 0.0 and overlapScore >= self.SDRThreshold[ entryIdx ]:
                self.columnSDRFits = True

                fittingEntries.append( entryIdx )
                fittingScores.append( vectorScore * overlapScore )
        # If multiple entries fit choose the one with the highest score.
        if self.columnSDRFits:
            self.thisEntryIndex = fittingEntries[ max( range( len( fittingScores ) ), key = fittingScores.__getitem__ ) ]

            # Change the cell entries.
#            if len( OCellSDR ) >= self.vectorMemoryDict[ "objectRepActivation" ]:
#                self.entryFCellSDR[ self.thisEntryIndex ] = FCellSDR
#                self.entryOCellSDR                        = OCellSDR

        # If zero entry doesn't fit then create a new one.
        else:
            self.thisEntryIndex = len( self.entryColumnSDR )
            self.entryColumnSDR.append( columnSDR )
            self.entryFCellSDR.append( [] )
            self.unityFCellSDR.append( [] )
#            if len( OCellSDR ) >= self.vectorMemoryDict[ "objectRepActivation" ]:
#                self.entryFCellSDR.append( FCellSDR )
#                self.entryOCellSDR = OCellSDR
#            else:
#                self.entryFCellSDR.append( [] )
#                self.entryOCellSDR = []
            self.entryCenter.append( [ self.currentPosition[ d ] for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ) ] )
            self.entryTime.append( 0 )
#            self.unityCellSDR.append( [] )
            self.entryCount.append( 1 )
            self.standardDeviation.append( [ self.vectorMemoryDict[ "initialStandardDeviation" ] ] * self.vectorMemoryDict[ "vectorDimensions" ] )
            self.vectConfidence.append( [ self.vectorMemoryDict[ "initialVectorConfidence" ] ] * self.vectorMemoryDict[ "vectorDimensions" ] )
            self.SDRThreshold.append( self.vectorMemoryDict[ "FActivationThresholdMin" ] )

        # Check if reached stability.
        self.reachedStability = True
        if not self.columnSDRFits:
            self.reachedStability = False
        else:
            numStable = 0
            for entry in self.entryFCellSDR:
                if len( entry ) >= self.vectorMemoryDict[ "WMEntrySize" ]:
                    numStable += 1
            if numStable / len( self.entryColumnSDR ) < self.vectorMemoryDict[ "WMStabilityPct" ]:
                self.reachedStability = False

        # Update unityCellSDR.
        self.unityCellSDR = []
        for entry in self.entryFCellSDR:
            if len( entry ) > 0:
                self.unityCellSDR.append( GenerateUnitySDR( entry, len( columnSDR ), self.vectorMemoryDict[ "cellsPerColumn" ] ) )
            else:
                self.unityCellSDR.append( [] )

#        self.UpdateEntries()

    def UpdateEntries( self, winnerCells ):
    # If any item in working memory is above threshold time steps then remove it.
    # Add the new active Fcells to working memory at the 0-vector location, append it on the end, if columnSDRFits.
    # If columnSDRFits then adjust the averages of the various vector and SDR thresholds.

        entryToDelete = []

        # Insert winner cells into entry SDR.
        if self.thisEntryIndex != None:
            self.entryFCellSDR[ self.thisEntryIndex ].append( winnerCells )

            if self.columnSDRFits:
                self.entryTime[ self.thisEntryIndex ] = -1
            for entryIdx in range( len( self.entryColumnSDR ) ):
                # Update time step.
                self.entryTime[ entryIdx ] += 1

                if self.entryTime[ entryIdx ] > self.vectorMemoryDict[ "WMEntryDecay" ]:
                    NoRepeatInsort( entryToDelete, entryIdx )

        # For any entry with entryCellSDR greater than threshold delete the oldest ones.
        for entry in self.entryFCellSDR:
            while len( entry ) > self.vectorMemoryDict[ "WMEntrySize" ]:
                del entry[ 0 ]

        # Delete items whose timeStep is above threshold.
        if len( entryToDelete ) > 0:
            for toDel in reversed( entryToDelete ):
                del self.entryColumnSDR[ toDel ]
                del self.entryFCellSDR[ toDel ]
                del self.unityFCellSDR[ toDel ]
#                self.entryOCellSDR = []
                del self.entryCenter[ toDel ]
                del self.entryTime[ toDel ]
                del self.entryCount[ toDel ]
                del self.standardDeviation[ toDel ]
                del self.vectConfidence[ toDel ]
                del self.SDRThreshold[ toDel ]
                if self.thisEntryIndex > toDel:
                    self.thisEntryIndex -= 1

    def UpdateAverages( self, incomingAverages ):
    # Updates the presently active entry with the incoming averages for active segments, using a weighted average.

        if self.thisEntryIndex != None:
            newEntryCount = self.entryCount[ self.thisEntryIndex ] + incomingAverages[ 0 ]

            if newEntryCount != 0:
                for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
                    # Update average for this entries distance scalar.
                    oldStandardDeviationWeight = self.standardDeviation[ self.thisEntryIndex ][ d ] * self.entryCount[ self.thisEntryIndex ]
                    self.standardDeviation[ self.thisEntryIndex ][ d ] = ( oldStandardDeviationWeight + incomingAverages[ 1 ][ d ] ) / newEntryCount

                    # Update average for this entries vector confidence.
                    oldVectorConfidenceWeight = self.vectConfidence[ self.thisEntryIndex ][ d ] * self.entryCount[ self.thisEntryIndex ]
                    self.vectConfidence[ self.thisEntryIndex ][ d ] = ( oldVectorConfidenceWeight + incomingAverages[ 2 ][ d ] ) / newEntryCount

                    # Update average for this entries vector center.
                    oldCenterWeightD = self.entryCenter[ self.thisEntryIndex ][ d ] * self.entryCount[ self.thisEntryIndex ]
                    newCenterWeightD = self.currentPosition[ d ] * incomingAverages[ 0 ]
                    self.entryCenter[ self.thisEntryIndex ][ d ] = ( oldCenterWeightD + newCenterWeightD ) / newEntryCount

                # Update average for this entries SDR threshold.
                oldSDRThresholdWeight = self.SDRThreshold[ self.thisEntryIndex ] * self.entryCount[ self.thisEntryIndex ]
                self.SDRThreshold[ self.thisEntryIndex ] = ( oldSDRThresholdWeight + incomingAverages[ 3 ] ) / newEntryCount

                # Update entries segment count.
                self.entryCount[ self.thisEntryIndex ] += newEntryCount

    def GetCellForColumn( self, col ):
    # For the predicted entry and given column return the cell working memory predicted, if it does.
    # If it didn't predict this column return None.

        for cell in range( col * self.vectorMemoryDict[ "cellsPerColumn" ], ( col * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
            if self.thisEntryIndex != None:
                if BinarySearch( self.unityCellSDR[ self.thisEntryIndex ], cell ):
                    return cell
        return None

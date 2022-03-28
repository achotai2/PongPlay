from bisect import bisect_left
from random import uniform, choice, sample, randrange, shuffle
from math import sqrt
from collections import Counter
from useful_functions import BinarySearch, NoRepeatInsort, ModThisSynapse, IndexIfItsIn, FastIntersect, GenerateUnitySDR

class Segment:

    def __init__( self, dimensions, incidentCellsList, incidentColumns, terminalCell, vector, initialActivationThreshold,
        vectorDistanceScaler, vectorConfidence ):
    # Initialize the inidividual segment structure class.

        self.active              = True                         # True means it's predictive, and above threshold terminal cells fired.
        self.stable              = True
        self.timeSinceActive     = 0                            # Time steps since this segment was active last.
        self.activationThreshold = initialActivationThreshold   # Minimum overlap required to activate segment.
        self.activeAboveThresh   = False
        self.activeInside        = False

        # Synapse portion.
        self.incidentSynapses    = incidentCellsList.copy()
        self.incidentColumns     = incidentColumns.copy()
        self.terminalSynapse     = terminalCell

        self.incidentActivation  = []           # A list of all cells that last overlapped with incidentSynapses.

        # Vector portion.
        self.dimensions       = dimensions                  # Number of dimensions vector positions will be in.
        self.distanceScaler   = vectorDistanceScaler        # The stretch on the distance scale, a value between 0.0 and 1.0. Smaller # stretches further.
        self.vectorConfidence = vectorConfidence            # The strength of confidence in the vector, a value between 0.0 and 1.0. Smaller # means less amplitude.
        self.vectorCenter     = vector                      # The origin of the vector.

        if len( vector ) != dimensions:
            print( "Vector sent to create segment not of same dimensions sent." )
            exit()

    def Inside( self, vector ):
    # Checks if given vector position is inside range. Calculates this based on a score between 0.0 and 1.0, which is used as a probability.

        # Calculate vector probability.
        probabilityScore = self.ReturnVectorScore( vector )
        randomScore = uniform( 0, 1 )

        if probabilityScore >= randomScore:
            return True
        else:
            return False

    def ReturnVectorScore( self, vector ):
    # Returns the score for a given vector position.

        if len( vector ) != self.dimensions:
            print( "Vector sent to ReturnVectorScore() not of same dimensions as Segment." )
            exit()

        # Calculate the distance from this vector to our self.vectorCenter.
        sum = 0
        for i in range( self.dimensions ):
            sum += ( self.vectorCenter[ i ] - vector[ i ] ) ** 2
        distance = sqrt( sum )

        return -1 * ( self.distanceScaler * distance ) ** 2 + self.vectorConfidence

    def IncidentCellActive( self, incidentCell ):
    # Takes the incident cell and adds it to incidentActivation.

        NoRepeatInsort( self.incidentActivation, incidentCell )

    def CheckColumnOverlapAndVector( self, columnSDR, vector ):
    # Checks the overlap of columnSDR if above threshold, and vector is inside.
    # If so make myself active. Return my overlap.

        if self.Inside( vector ):
            self.activeInside = True

            overlap = len( FastIntersect( self.incidentColumns, columnSDR ) )

            if overlap >= self.activationThreshold:
                self.stable          = True
                return self.incidentSynapses
            else:
                self.stable          = False
        else:
            self.activeInside = False

        return None

    def CheckActivation( self, vector ):
    # Checks the incidentActivation against activationThreshold to see if segment becomes active.

        if len( self.incidentActivation ) >= self.activationThreshold:
            self.activeAboveThresh = True

            if self.activeInside or self.Inside( vector ):
                self.active          = True
                self.timeSinceActive = 0
                return True

        self.active = False
        return False

    def RefreshSegment( self ):
    # Updates or refreshes the state of the segment.

        self.active             = False
        self.incidentActivation = []
        self.timeSinceActive   += 1
        self.activeAboveThresh  = False

    def RemoveIncidentSynapse( self, synapseToDelete, columnToDelete ):
    # Delete the incident synapse sent.

        index = IndexIfItsIn( self.incidentSynapses, synapseToDelete )
        if index != None:
            del self.incidentSynapses[ index ]
        else:
            print( "Attempt to remove synapse from segment, but synapse doesn't exist." )
            exit()

        if self.incidentColumns[ index ] == columnToDelete:
            del self.incidentColumns[ index ]
        else:
            print( "Column doesn't match cell." )
            exit()

    def NewIncidentSynapse( self, synapseToCreate, columnToCreate ):
    # Create a new incident synapse.

        index = bisect_left( self.incidentSynapses, synapseToCreate )

        if index == len( self.incidentSynapses ):
            self.incidentSynapses.append( synapseToCreate )
            self.incidentColumns.append( columnToCreate )
        elif self.incidentSynapses[ index ] != synapseToCreate:
            self.incidentSynapses.insert( index, synapseToCreate )
            self.incidentColumns.insert( index, columnToCreate )
        else:
            print( "Attempt to add synapse to segment, but synapse already exists." )
            exit()

    def AlreadySynapseToColumn( self, checkSynapse, cellsPerColumn ):
    # Checks if this segment has a synapse to the same column as checkSynapse.

        checkColumn = int( checkSynapse / cellsPerColumn )

        if BinarySearch( self.incidentColumns, checkColumn ):
            return True
        else:
            return False

    def Equality( self, other, equalityThreshold ):
    # Runs the following comparison of equality: segment1 == segment2, comparing their activation intersection, but not vector.

        if self.terminalSynapse == other.terminalSynapse and len( FastIntersect( self.incidentSynapses, other.incidentSynapses ) ) > equalityThreshold:
            return True
        else:
            return False

    def AdjustThreshold( self, minActivation, maxActivation ):
    # Check the number of bundles that have selected winner. Use this to adjust activationThreshold.

        numWinners = 0

        for incB in self.incidentPermanences:
            if incB == 1.0:
                numWinners += 1

        self.activationThreshold = ( ( minActivation - maxActivation ) * 2 ** ( -1 * numWinners ) ) + maxActivation

    def CellForColumn( self, column ):
    # Return the cell for the column.

        index = IndexIfItsIn( self.incidentColumns, column )
        if index != None:
            return self.incidentSynapses[ index ]
        else:
            return None

#-------------------------------------------------------------------------------

class SegmentStructure:

    def __init__( self, vectorDimensions, initialPermanence, permanenceIncrement, permanenceDecrement, permanenceDecay, activeThresholdMin,
        activeThresholdMax, incidentColumnDimensions, incidentCellsPerColumn, maxSynapsesToAddPer, maxSynapsesPerSegment, maxTimeSinceActive,
        equalityThreshold, initVectorScaleFactor, initVectorConfidence ):
    # Initialize the segment storage and handling class.

        self.dimensions            = vectorDimensions
        self.initialPermanence     = initialPermanence
        self.permanenceIncrement   = permanenceIncrement
        self.permanenceDecrement   = permanenceDecrement
        self.permanenceDecay       = permanenceDecay
        self.activeThresholdMin    = activeThresholdMin
        self.activeThresholdMax    = activeThresholdMax
        self.cellsPerColumn        = incidentCellsPerColumn
        self.columnDimensions      = incidentColumnDimensions
        self.maxSynapsesToAddPer   = maxSynapsesToAddPer
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.maxTimeSinceActive    = maxTimeSinceActive
        self.equalityThreshold     = equalityThreshold
        self.initVectorScaleFactor = initVectorScaleFactor
        self.initVectorConfidence  = initVectorConfidence

        self.segments       = []                                  # Stores all segments structures.
        self.activeSegments = []

        self.bestCells      = [ [] for i in range( incidentColumnDimensions ) ]

        self.segsToDelete   = []

        self.incidentSegments = []                          # Stores the connections from each incident cell to segment.
        for cell in range( incidentCellsPerColumn * incidentColumnDimensions ):
            self.incidentSegments.append( [] )
        self.incidentPermanences = []
        for cell in range( incidentCellsPerColumn * incidentColumnDimensions ):
            self.incidentPermanences.append( [] )

        # Working Memory portion.-----------------------------------------------
        self.numPositionCells   = 100
        self.maxPositionRange   = 800
        self.positionCellScores = [ [ [ [], [], [], [] ] for i in range( incidentCellsPerColumn ) ] for j in range( incidentColumnDimensions) ]
        self.currentPosition    = []
        for d in range( vectorDimensions ):
            self.currentPosition.append( 0 )

    def HowManyActiveSegs( self ):
    # Return the number of active segments.

        return len( self.activeSegments )

    def AddSynapse( self, incCell, segIndex ):
    # Add a synapse to specified segment.

        insertIndex = bisect_left( self.incidentSegments[ incCell ], segIndex )

        if insertIndex != len( self.incidentSegments[ incCell ] ) and self.incidentSegments[ incCell ][ insertIndex ] == segIndex:
            print( "Synapse to this segment already exists." )
            exit()

        self.incidentSegments[ incCell ].insert( insertIndex, segIndex )
        self.incidentPermanences[ incCell ].insert( insertIndex, self.initialPermanence )
        self.segments[ segIndex ].NewIncidentSynapse( incCell, int( incCell / self.cellsPerColumn ) )

    def DeleteSynapse( self, incCell, segIndex ):
    # Remove a synapse to specified segment.

        # Check if synapse exists and if so delete it and all references to it.
        delIndex = bisect_left( self.incidentSegments[ incCell ], segIndex )

        if delIndex != len( self.incidentSegments[ incCell ] ) and self.incidentSegments[ incCell ][ delIndex ] == segIndex:
            del self.incidentSegments[ incCell ][ delIndex ]
            del self.incidentPermanences[ incCell ][ delIndex ]
            self.segments[ segIndex ].RemoveIncidentSynapse( incCell, int( incCell / self.cellsPerColumn ) )

        # Check if segment has any synapses left. If none then mark segment for deletion.
        if len( self.segments[ segIndex ].incidentSynapses ) == 0:
            self.segsToDelete.append( segIndex )

        return delIndex

    def DeleteSegments( self, Cells ):
    # Receives a list of indices of segments that need deletion. Deletes these segments from self.segments,
    # and removes all references to them in self.incidentSegments.

        if len( self.segsToDelete ) > 0:

            self.segsToDelete.sort()

            for segIndex in reversed( self.segsToDelete ):
                # Delete any references to segment, and lower the index of all greater segment reference indices by one.
                for incCell, incList in enumerate( self.incidentSegments ):
                    indexAt = self.DeleteSynapse( incCell, segIndex )

                    while indexAt < len( incList ):
                        incList[ indexAt ] -= 1
                        indexAt += 1

                # Decrease terminal reference.
                if self.segments[ segIndex ].terminalSynapse != None:
                    Cells[ self.segments[ segIndex ].terminalSynapse ].isTerminalCell -= 1

                # Delete any references to this segment if they exist, and modify indices.
                actIndex = bisect_left( self.activeSegments, segIndex )
                if actIndex != len( self.activeSegments ) and self.activeSegments[ actIndex ] == segIndex:
                    del self.activeSegments[ actIndex ]
                while actIndex < len( self.activeSegments ):
                    self.activeSegments[ actIndex ] -= 1
                    actIndex += 1

                # Delete references to this segment in self.positionCellScores.
                terminalCell = int( self.segments[ segIndex ].terminalSynapse % self.cellsPerColumn )
                terminalCol  = int( self.segments[ segIndex ].terminalSynapse / self.cellsPerColumn )
                posIndex = bisect_left( self.positionCellScores[ terminalCol ][ terminalCell ][ 0 ], segIndex )
                if ( posIndex != len( self.positionCellScores[ terminalCol ][ terminalCell ][ 0 ] ) and
                    self.positionCellScores[ terminalCol ][ terminalCell ][ 0 ][ posIndex ] == segIndex ):
                    del self.positionCellScores[ terminalCol ][ terminalCell ][ 0 ][ posIndex ]
                    del self.positionCellScores[ terminalCol ][ terminalCell ][ 1 ][ posIndex ]
                    del self.positionCellScores[ terminalCol ][ terminalCell ][ 2 ][ posIndex ]
                # Modify all segment indices greater than to be minus one.
                for column in range( self.columnDimensions ):
                    for cell in range( self.cellsPerColumn ):
                        for entry in range( len( self.positionCellScores[ column ][ cell ][ 0 ] ) ):
                            if self.positionCellScores[ column ][ cell ][ 0 ][ entry ] > segIndex:
                                self.positionCellScores[ column ][ cell ][ 0 ][ entry ] -= 1

                # Delete the segment.
                del self.segments[ segIndex ]

        self.segsToDelete = []

    def CreateSegment( self, Cells, incidentCellsList, terminalCell, vector ):
    # Creates a new segment.

        newSegment = Segment( self.dimensions, [], [], terminalCell, vector, self.activeThresholdMin, self.initVectorScaleFactor, self.initVectorConfidence )
        self.segments.append( newSegment )

        indexOfNew = len( self.segments ) - 1
        self.activeSegments.append( indexOfNew )

        for incCell in incidentCellsList:
            self.AddSynapse( incCell, indexOfNew )

        if terminalCell != None:
            Cells[ terminalCell ].isTerminalCell += 1

    def UnifyVectors( self, vectorCellsList ):
    # Use the list of vector cells contained in vectorCellsList and unify them into one by taking the max for each cell.

        for entry in vectorCellsList:
            if len( entry ) != self.dimensions:
                print( "Dimensions of vectors sent to UnifyVectors() not of correct size." )
                exit()
            for cellList in entry:
                if len( cellList ) != self.numVectorSynapses:
                    print( "Number of cells in vectors sent to UnifyVectors() not of correct size." )
                    exit()

        newVectorCells = []
        for dim in range( self.dimensions ):
            newPermanences = []
            for cell in range( self.numVectorSynapses ):
                max = 0.0
                for entry in vectorCellsList:
                    if entry[ dim ][ cell ] > max:
                        max = entry[ dim ][ cell ]

                newPermanences.append( max )
            newVectorCells.append( newPermanences )

        return newVectorCells

    def UpdateSegmentActivity( self ):
    # Make every segment that was active inactive, and refreshes its synapse activation.
    # Also add a time step to each segment, and see if it dies as a result. Delete any segments that die.

        self.activeSegments = []

        for index, segment in enumerate( self.segments ):
            segment.RefreshSegment()
            if segment.timeSinceActive > self.maxTimeSinceActive:
                self.segsToDelete.append( index )

    def SegmentLearning( self, Cells, lastWinnerCells, lastActiveCells, doDecayCreate ):
    # Perform learning on all active and inactive segments.
    # Refresh all segments then perform learning on them.
    # Delete segments that need deleting.

        if doDecayCreate:
            self.DecayAndCreate( Cells, lastWinnerCells, lastActiveCells )

#        for actSeg in self.activeSegments:
#            segment.AdjustThreshold( self.activeThresholdMin, self.activeThresholdMax )

        self.CheckIfSegsIdentical( Cells )

        self.UpdateSegmentActivity()

        self.DeleteSegments( Cells )

# CAN PROBABLY REMOVE THE BELOW FUNCTION.............................
    def UpdateStableSegments( self, columnSDR, vector ):
    # Update the relativePosition of all stable segments. If any relativePosition is Inside then get its incident cells.
    # Use these incident cells to form bestCells list.

        # Update all the stable segments relativePositions, and check their relativePosition is Inside.
        # If inside then check its column overlap to see if it stays stable.
        # Also generate the columns list of incident cells.
        columnsList = [ [] for i in range( self.columnDimensions ) ]

        noLongerStable = []
        for staIndex, stableSeg in enumerate( self.stableSegments ):
            self.segments[ stableSeg ].UpdateRelativePosition( vector )
            thisSegsIncident = self.segments[ stableSeg ].CheckColumnOverlapAndVector( columnSDR, vector )

            if thisSegsIncident == None:
                noLongerStable.append( staIndex )

            # Add the returned incident cells list to the columns list.
            else:
                for incCell in thisSegsIncident:
                    NoRepeatInsort( columnsList[ int( incCell / self.cellsPerColumn ) ], incCell )

        for s in reversed( noLongerStable ):
            del self.stableSegments[ s ]

        # Make the bestCells list, choosing for each column the cell that appears most - if tied then choose randomly.
        self.bestCells = [ [] for i in range( self.columnDimensions ) ]

        for colIndex, col in enumerate( columnsList ):
            if len( col ) > 0:
                occuranceCount = Counter( col )
                frequencyList = occuranceCount.most_common()

                randomChoices = []
                for index, cell in enumerate( frequencyList ):
                    if index == 0:
                        randomChoices.append( cell[ 0 ] )
                    else:
                        if cell[ 1 ] == frequencyList[ 0 ][ 1 ]:
                            randomChoices.append( cell[ 0 ] )
                self.bestCells[ colIndex ].append( choice( randomChoices ) )

    def GetStimulatedSegments( self, activeCells, vector ):
    # Using the activeCells and vector find all segments that activate. Add these segments to a list and return it.

        if len( self.incidentSegments ) > 0:
            # Activate the synapses in segments using activeCells.
            for incCell in activeCells:
                for entry in self.incidentSegments[ incCell ]:
                    self.segments[ entry ].IncidentCellActive( incCell )

            # Check the overlap of all segments and see which ones are active, and add the terminalCell to stimulatedCells.
            for segIndex, segment in enumerate( self.segments ):
                if segment.CheckActivation( vector ):
                    NoRepeatInsort( self.activeSegments, segIndex )

        predictedCells = []
        for actSeg in self.activeSegments:
            NoRepeatInsort( predictedCells, self.segments[ actSeg ].terminalSynapse )

        return predictedCells

    def UpdateVector( self, vector ):
    # Use the vector to update the local workingMemory position.

        if len( vector ) != self.dimensions:
            print( "Vectors sent to working memory of wrong size." )
            exit()

        # Update vector.
        for d in range( self.dimensions ):
            self.currentPosition[ d ] += vector[ d ]

# CAN REMOVE BELOW FUNCTION I THINK...............................
    def GetIndexFromPosition( self, coordinates ):
    # Given the position calculate and return the indices for our position list.

        screenSize = self.vectorRange * 0.5

        indices = []

        for i in range( self.dimensions ):
            indices.append( int( ( ( coordinates[ i ] + ( screenSize * 0.5 ) ) / screenSize ) * self.numPositionCells ) )

        return indices

# CAN REMOVE BELOW FUNCTION I THINK...............................
    def GetPositionFromIndex( self, indices ):
    # Given self.positionCellScores indices, get us the approximate screen position.

        coordinates = []

        for i in range( self.dimensions ):
            coordinates.append( int( ( ( indices[ i ] / self.numPositionCells ) * self.maxPositionRange ) - ( self.maxPositionRange * 0.5 ) ) )

        return coordinates

    def UpdateWorkingMemory( self, vector ):
    # Given this time steps active segments, update self.positionCellScores.

        # Go through every active segment and get their scores.
        for actSeg in self.activeSegments:
            terminalCell = int( self.segments[ actSeg ].terminalSynapse % self.cellsPerColumn )
            terminalCol  = int( self.segments[ actSeg ].terminalSynapse / self.cellsPerColumn )
            overlapScore = len( self.segments[ actSeg ].incidentActivation )

            index = bisect_left( self.positionCellScores[ terminalCol ][ terminalCell ][ 0 ], actSeg )

            if index == len( self.positionCellScores[ terminalCol ][ terminalCell ][ 0 ] ):
                self.positionCellScores[ terminalCol ][ terminalCell ][ 0 ].append( actSeg )
                self.positionCellScores[ terminalCol ][ terminalCell ][ 1 ].append( 1 )
                self.positionCellScores[ terminalCol ][ terminalCell ][ 2 ].append( overlapScore )
#                self.positionCellScores[ terminalCol ][ terminalCell ][ 2 ].append( 0.5 )
                self.positionCellScores[ terminalCol ][ terminalCell ][ 3 ].append( self.currentPosition.copy() )
            elif self.positionCellScores[ terminalCol ][ terminalCell ][ 0 ][ index ] == actSeg:
                self.positionCellScores[ terminalCol ][ terminalCell ][ 1 ][ index ] += 1
                self.positionCellScores[ terminalCol ][ terminalCell ][ 2 ][ index ] += overlapScore
                for d in range( self.dimensions ):
                    self.positionCellScores[ terminalCol ][ terminalCell ][ 3 ][ index ][ d ] += self.currentPosition[ d ]
            else:
                self.positionCellScores[ terminalCol ][ terminalCell ][ 0 ].insert( index, actSeg )
                self.positionCellScores[ terminalCol ][ terminalCell ][ 1 ].insert( index, 1 )
                self.positionCellScores[ terminalCol ][ terminalCell ][ 2 ].insert( index, overlapScore )
#                self.positionCellScores[ terminalCol ][ terminalCell ][ 2 ].insert( index, 0.5 )
                self.positionCellScores[ terminalCol ][ terminalCell ][ 3 ].insert( index, self.currentPosition.copy() )

        self.UpdateVector( vector )

    def GetWinnerCellForColumn( self, column ):
    # Given the current position, and given column, find each cells highest score.
    # Sort the cells in the column by score and return this sorted list (from highest score to lowest).

        cellList   = [ i for i in range( self.cellsPerColumn ) ]
        shuffle( cellList )
        cellScores = [ 0.0 ] * self.cellsPerColumn
        cellOverlap = [ None ] * self.cellsPerColumn

        # Go through the cells in this column in self.positionCellScores.
        for cellIndex, cell in enumerate( cellList ):
            thisEntry = self.positionCellScores[ column ][ cell ]

            for seg in range( len( thisEntry[ 1 ] ) ):
                segIndex       = thisEntry[ 1 ][ seg ]

                avgVector      = []
                for d in range( self.dimensions ):
                    avgIncidentPosD = thisEntry[ 3 ][ seg ][ d ] / thisEntry[ 1 ][ seg ]
                    avgVector.append( self.currentPosition[ d ] - avgIncidentPosD )

                thisVectorScore = self.segments[ segIndex ].ReturnVectorScore( avgVector )
#                overlapAverage = thisEntry[ 2 ][ seg ] / thisEntry[ 1 ][ seg ]
#                overlapAverage = self.ReturnActivation( segIndex )
                overlapAverage = 1

                # Calculate the score for this segment.
                thisScore = thisVectorScore * overlapAverage

                if thisScore > cellScores[ cellIndex ]:
                    cellScores[ cellIndex ] = thisScore
                    cellOverlap[ cellIndex ] = segIndex

        # Sort the cells by score.
        sortedCellsList = []
        sortedSegList   = []
        while len( cellList ) > 0:
            winnerIndex  = 0
            highestScore = cellScores[ 0 ]
            for celI, sco in enumerate( cellScores ):
                if sco > highestScore:
                    winnerIndex  = celI
                    highestScore = sco
            sortedCellsList.append( cellList[ winnerIndex ] + ( column * self.cellsPerColumn ) )
            sortedSegList.append( winnerIndex )
            del cellList[ winnerIndex ]
            del cellScores[ winnerIndex ]

        # Support the score for the winner, and decrease others.
#        if cellOverlap[ sortedSegList[ 0 ] ] != None:
#            self.segments[ cellOverlap[ sortedSegList[ 0 ] ] ].vectorConfidence += 0.05
#        for c in range( 1, self.cellsPerColumn ):
#            if cellOverlap[ sortedSegList[ c ] ] != None:
#                self.segments[ cellOverlap[ sortedSegList[ c ] ] ].vectorConfidence -= 0.05

        return sortedCellsList

# CAN REMOVE BELOW FUNCTION I THINK...............................
    def CalculatePredictedScores( self, vector ):
    # Using the active segments, gather their terminal cells and return it.
    # Also, for each such terminal cell, calculate its position scores, insert it into the list, and return this.


        allPosScores = []
        for x in range( self.numPositionCells ):
            allPosScores.append( [] )
            for y in range( self.numPositionCells ):
                allPosScores[ x ].append( [] )
                for c in range( self.numPositionCells ):
                    thisPosition = self.GetPositionFromIndex( [ x, y, c ] )
                    thisVector   = []
                    for i in range( self.dimensions ):
                        thisVector.append( thisPosition[ i ] - self.currentPosition[ i ] )

                    segPosScore = self.segments[ actSeg ].ReturnVectorScore( thisVector )
                    allPosScores[ x ][ y ].append( segPosScore )


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






        for actSeg in self.activeSegments:
            terminalCell = self.segments[ actSeg ].terminalSynapse

            # Calculate the score.
            overlapScore  = len( self.segments[ actSeg ].incidentActivation )
            vectorCells   = self.segments[ actSeg ].ReturnVectorCells()
            cellScore     = [ [] for d in range( self.dimensions ) ]
            for d in range( self.dimensions ):
                for x in range( self.numVectorSynapses ):
                    cellScore[ d ].append( overlapScore * vectorCells[ d ][ x ] )

            # Add the terminal cell.
            cellIndex = bisect_left( terminalCellsList, terminalCell )
            if cellIndex == len( terminalCellsList ):
                terminalCellsList.append( terminalCell )

                columnScoreList.append( [ terminalCell, 1, cellScore ] )

            elif terminalCellsList[ cellIndex ] != terminalCell:
                terminalCellsList.insert( cellIndex, terminalCell )

                columnScoreList.insert( cellIndex, [ terminalCell, 1, cellScore ] )

            else:
                columnScoreList[ cellIndex ][ 1 ] += 1

                for d in range( self.dimensions ):
                    for x in range( self.numVectorSynapses ):
                        columnScoreList[ cellIndex ][ 2 ][ d ][ x ] += cellScore[ d ][ x ]

        return terminalCellsList, columnScoreList

# CAN REMOVE BELOW FUNCTION I THINK...............................
    def GetCellForColumn( self, column ):
    # Returns the cell for the given column from self.bestCells.

        if len( self.bestCells[ column ] ) > 0:
            return self.bestCells[ column ][ 0 ]
        else:
            return None

# CAN REMOVE BELOW FUNCTION I THINK...............................
    def CheckSegmentColumnOverlap( self, columnSDR, vector ):
    # Checks all segments if their columns overlap with columnSDR above threshold.
    # Also check if their vector activates. Those that do make them active.

        mostSegment = None
        mostOverlap = 0

        for segIndex, segment in enumerate( self.segments ):
            overlap = segment.CheckColumnOverlapAndVector( columnSDR, vector )
            if overlap != None and overlap > mostOverlap:
                mostSegment = segIndex

        if mostSegment != None:
            NoRepeatInsort( self.activeSegments, mostSegment )
        return mostSegment

# CAN PROBABLY REMOVE THE BELOW FUNCTION.............................
    def CellForColumn( self, segIndex, column ):
    # Return cell for column for indexed segment.

        return self.segments[ segIndex ].CellForColumn( column )

    def ChangePermanence( self, incCell, segIndex, permanenceChange ):
    # Change the permanence of synapse incident on incCell, part of segIndex, by permanenceChange.
    # If permanence == 0.0 then delete it.

        entryIndex = IndexIfItsIn( self.incidentSegments[ incCell ], segIndex )
        if entryIndex != None:
            self.incidentPermanences[ incCell ][ entryIndex ] = ModThisSynapse( self.incidentPermanences[ incCell ][ entryIndex ], permanenceChange, True )

            if self.incidentPermanences[ incCell ][ entryIndex ] <= 0.0:
                self.DeleteSynapse( incCell, segIndex )

    def DecayAndCreate( self, Cells, lastWinnerCells, lastActiveCells ):
    # For all active segments:
    # 1.) Decrease all synapses on active segments where the terminal cell is not a winner.
    # 2.) Increase synapse strength to active incident cells that already have synapses.
    # 3.) Decrease synapse strength to inactive incident cells that already have synapses.
    # 4.) Build new synapses to active incident winner cells that don't have synapses.

        # Modify the stable synapses.
#        for ssIndex, stableSeg in enumerate( self.stableSegments ):
#            if self.segments[ stableSeg ].activeInside and not Cells[ self.segments[ stableSeg ].terminalSynapse ].active:
#                self.segments[ stableSeg ].stable = False

#                for incCell in self.segments[ stableSeg ].incidentSynapses:
#                    self.ChangePermanence( incCell, stableSeg, -self.permanenceDecrement )

        # Then deal with active segments.
        for activeSeg in self.activeSegments:
            # 1.)...
            if self.segments[ activeSeg ].terminalSynapse != None and not Cells[ self.segments[ activeSeg ].terminalSynapse ].winner:
                for incCell in self.segments[ activeSeg ].incidentSynapses:
                    self.ChangePermanence( incCell, activeSeg, -self.permanenceDecrement )

            else:
                synapseToAdd = lastWinnerCells.copy()
                for incCell in self.segments[ activeSeg ].incidentSynapses:
                    # 2.)...
                    if BinarySearch( lastActiveCells, incCell ):
                        self.ChangePermanence( incCell, activeSeg, self.permanenceIncrement )
                    # 3.)...
                    else:
                        self.ChangePermanence( incCell, activeSeg, -self.permanenceDecrement )

                    indexIfIn = IndexIfItsIn( synapseToAdd, incCell )
                    if indexIfIn != None:
                        del synapseToAdd[ indexIfIn ]

                if len( synapseToAdd ) > 0:
                    # 4.)...
                    # Check to make sure this segment doesn't already have a synapse to this column.
                    realSynapsesToAdd = []
                    for synAdd in synapseToAdd:
                        if not self.segments[ activeSeg ].AlreadySynapseToColumn( synAdd, self.cellsPerColumn ):
                            realSynapsesToAdd.append( synAdd )

                    reallyRealSynapsesToAdd = sample( realSynapsesToAdd, min( len( realSynapsesToAdd ), self.maxSynapsesToAddPer ) )
                    for toAdd in reallyRealSynapsesToAdd:
                        self.AddSynapse( toAdd, activeSeg )

                    # If the number of synapses is above maxSynapsesPerSegment then delete random synapses.
                    while self.maxSynapsesPerSegment - len( self.segments[ activeSeg ].incidentSynapses ) < 0:
                        toDel = choice( self.segments[ activeSeg ].incidentSynapses )
                        self.ChangePermanence( toDel, activeSeg, -1.0 )

    def ReturnActivation( self, seg ):
    # Returns the sum of permanences of synapses incident on incCellList and part of seg.

        activation = 0.0

        for incCell in self.segments[ seg ].incidentSynapses:
            entryIndex = IndexIfItsIn( self.incidentSegments[ incCell ], seg )
            if entryIndex != None:
                activation += self.incidentPermanences[ incCell ][ entryIndex ]

        return activation

    def ThereCanBeOnlyOne( self, cellsList ):
    # Return the cell, in cellsList, which is terminal on an active segment with greatest activation.

        greatestActivation = 0.0
        greatestCell       = cellsList[ 0 ]

        if len( cellsList ) > 1:
            for cell in cellsList:
                for actSeg in self.activeSegments:
                    if self.segments[ actSeg ].terminalSynapse == cell:
                        thisActivation = self.ReturnActivation( actSeg )
                        if thisActivation > greatestActivation:
                            greatestActivation = thisActivation
                            greatestCell       = cell

        return ( greatestCell, greatestActivation )

    def CheckIfSegsIdentical( self, Cells ):
    # Compares all segments to see if they have identical vectors or active synapse bundles. If any do then merge them.
    # A.) Begin by checking which segments have an above threshold overlap activation.
    # B.) Group these segments by forming a list of lists, where each entry is a grouping of segments.
    #   Segments are grouped by first checking their terminal synapse against one, and then checking if their overlap is above equalityThreshold.
    # C.) Then, if any entry has more than two segments in it, merge the two, and mark one of the segments for deletion.

        segmentGroupings = []

        for index, segment in enumerate( self.segments ):
            if segment.activeAboveThresh:
                chosenIndex = None
                for entryIndex, entry in enumerate( segmentGroupings ):
                    thisEntryMatches = True
                    for entrySegment in entry:
                        if not segment.Equality( self.segments[ entrySegment ], self.equalityThreshold ):
                            thisEntryMatches = False
                            break
                    if thisEntryMatches:
                        chosenIndex = entryIndex
                        break

                if chosenIndex == None:
                    segmentGroupings.append( [ index ] )
                else:
                    segmentGroupings[ chosenIndex ].append( index )

# THIS COULD PROBABLY BE MADE BETTER BY MERGING THEM, RATHER THAN JUST DELETING ONE.
        for group in segmentGroupings:
            if len( group ) > 1:
#                SDRList     = [ self.segments[ group[ i ] ].incidentSynapses for i in range( len( group) ) ]
#                unitySDR    = GenerateUnitySDR( SDRList, len( SDRList[ 0 ] ), self.cellsPerColumn )

                winnerSegment = group.pop( randrange( len( group ) ) )

#                unityVector = self.UnifyVectors( [ self.segments[ group[ k ] ].ReturnVectorCells() for k in range( len( group ) ) ] )

#                self.CreateSegment( Cells, unitySDR, self.segments[ group[ 0 ] ].terminalSynapse, None )

                for segIndex in range( len( group ) ):
                    self.segsToDelete.append( group[ segIndex ] )

# ------------------------------------------------------------------------------

class FCell:

    def __init__( self ):
    # Create a new feature level cell with synapses to OCells.

        # FCell state variables.
        self.active     = False
        self.lastActive = False
        self.predicted  = False
        self.winner     = False
        self.lastWinner = False

        self.isTerminalCell = 0           # Number of segments this cell is terminal on.

        # For data collection.
        self.activeCount = 0
        self.states      = []
        self.statesCount = []

#    def __repr__( self ):
#    # Returns string properties of the FCell.
#
#        toPrint = []
#        for index in range( len( self.OCellConnections ) ):
#            toPrint.append( ( self.OCellConnections[ index ], self.OCellPermanences[ index ] ) )
#
#        return ( "< ( Connected OCells, Permanence ): %s >"
#            % toPrint )

    def ReceiveStateData( self, stateNumber ):
    # Receive the state number and record it along with the count times, if this cell is active.

        if self.active:
            self.activeCount += 1

            stateIndex = bisect_left( self.states, stateNumber )
            if stateIndex != len( self.states ) and self.states[ stateIndex ] == stateNumber:
                self.statesCount[ stateIndex ] += 1
            else:
                self.states.insert( stateIndex, stateNumber )
                self.statesCount.insert( stateIndex, 1 )

    def ReturnStateInformation( self ):
    # Return the state information and count.

        toReturn = []
        toReturn.append( self.activeCount )
        for i, s in enumerate( self.states ):
            toReturn.append( ( s, self.statesCount[ i ] ) )

        return toReturn

# ------------------------------------------------------------------------------

class OCell:

    def __init__( self ):
    # Create new object cell.

        self.active = False

        self.isTerminalCell = 0

        self.segActivationLevel     = 0
        self.overallActivationLevel = 0.0

    def AddActivation( self, synapseActivation ):
    # Add plust one to segActivationLevel, and add synapseActivation to the overallActivationLevel .

        self.segActivationLevel += 1
        self.overallActivationLevel += synapseActivation

    def ResetState( self ):
    # Make inactive and reset activationLevel.

        self.active = False

        self.segActivationLevel     = 0
        self.overallActivationLevel = 0.0

    def CheckSegmentActivationLevel( self, threshold ):
    # Returns True if segActivationLevel is above threshold.

        if self.segActivationLevel >= threshold:
            return True
        else:
            return False

    def ReturnOverallActivation( self ):
    # Returns self.overallActivationLevel.

        return self.overallActivationLevel

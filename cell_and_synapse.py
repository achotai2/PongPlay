from bisect import bisect_left
from random import uniform, choice, sample, randrange, shuffle
from collections import Counter
from useful_functions import BinarySearch, NoRepeatInsort, ModThisSynapse, IndexIfItsIn, FastIntersect, GenerateUnitySDR, CalculateDistanceScore
import numpy
import math
#from time import time

class Segment:

    def __init__( self, vectorMemoryDict, terminalCell, vector, objectCellsList ):
    # Initialize the inidividual segment structure class.

        self.vectorMemoryDict = vectorMemoryDict

        self.active              = False                         # True means it's predictive, and above threshold terminal cells fired.
        self.stimulated          = 0.0
        self.timeSinceActive     = 0                            # Time steps since this segment was active last.
        self.activationThreshold = vectorMemoryDict[ "FActivationThresholdMin" ]   # Minimum overlap required to activate segment.
        self.activeAboveThresh   = False

        # Lateral synapse portion.
        self.incidentSynapses    = []
        self.incidentColumns     = []
        self.terminalSynapse     = terminalCell

        self.incidentActivation  = []           # A list of all cells that last overlapped with incidentSynapses.

        # Vector portion.
        self.vectorCenter        = vector                      # The origin of the vector.
        self.standardDeviation   = [ self.vectorMemoryDict[ "initialStandardDeviation" ] ] * self.vectorMemoryDict[ "vectorDimensions" ]
        self.vectorConfidence    = [ self.vectorMemoryDict[ "initialVectorConfidence" ] ] * self.vectorMemoryDict[ "vectorDimensions" ]

        self.lastProbabilityScore = 0
        self.segmentConfidence    = 0.0
        self.lastVector = [ 0 ] * self.vectorMemoryDict[ "vectorDimensions" ]

        self.vectorRange       = []
        for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
            self.vectorRange.append( [ -self.vectorMemoryDict[ "vectorRange" ], self.vectorMemoryDict[ "vectorRange" ] ] )

        if vector != None:
            if len( vector ) != self.vectorMemoryDict[ "vectorDimensions" ]:
                print( "Vector sent to create segment not of same dimensions sent." )
                exit()

            self.vectorSynapses = numpy.zeros( shape = [ self.vectorMemoryDict[ "numVectorSynapses" ] ] * self.vectorMemoryDict[ "vectorDimensions" ] )
# MIGHT NEED TO ADD A SCALE FUNCTION, AND RANGE FUNCTION INTO THIS LATER.
#            for y in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
#                self.ChangeVectorStrength( y, vector[ y ], self.vectorMemoryDict[ "initialPermanence" ] )
        else:
            self.vectorSynapses = vectorCells

        # Object synapse portion.
        self.OCellSynapses    = objectCellsList.copy()
        self.OCellPermanences = [ self.vectorMemoryDict[ "initialPermanence" ] ] * len( objectCellsList )
        self.OCellOVerlap     = False

        if len( vector ) != self.vectorMemoryDict[ "vectorDimensions" ]:
            print( "Vector sent to create segment not of same dimensions sent." )
            exit()

    def OCellLearning( self, OCells, activeOCells, initialPermanence, permanenceIncrement, permanenceDecrement, maxToAdd, maxSynapses ):
    # Support synapses to activeOCells, and create new if don't exist. Decay synapses to non-activeOCells.

        synapseToAdd    = activeOCells.copy()
        synapseToDelete = []

        for synIndex, syn in enumerate( self.OCellSynapses ):
            if not OCells[ syn ].active:
                self.OCellPermanences[ synIndex ] = ModThisSynapse( self.OCellPermanences[ synIndex ], -permanenceDecrement, 1.0, 0.0, True )
                if self.OCellPermanences[ synIndex ] <= 0.0:
                    synapseToDelete.append( synIndex )
            else:
                self.OCellPermanences[ synIndex ] = ModThisSynapse( self.OCellPermanences[ synIndex ], permanenceIncrement, 1.0, 0.0, True )
                i = IndexIfItsIn( synapseToAdd, syn )
                if i != None:
                    del synapseToAdd[ i ]
                else:
                    print( "Active OCell missing from activeOCells in OCellLearning()" )
                    exit()

        if len( synapseToDelete ) > 0:
            for toDel in reversed( synapseToDelete ):
                del self.OCellSynapses[ toDel ]
                del self.OCellPermanences[ toDel ]

        if len( synapseToAdd ) > 0:
            numAdded = 0
            while len( self.OCellSynapses ) <= maxSynapses and numAdded <= maxToAdd and len( synapseToAdd ) > 0:
                index = randrange( len( synapseToAdd ) )
                insertion = bisect_left( self.OCellSynapses, synapseToAdd[ index ] )
                if insertion == len( self.OCellSynapses ):
                    self.OCellSynapses.append( synapseToAdd[ index ] )
                    self.OCellPermanences.append( initialPermanence )
                else:
                    self.OCellSynapses.insert( insertion, synapseToAdd[ index ] )
                    self.OCellPermanences.insert( insertion, initialPermanence )

                numAdded += 1
                del synapseToAdd[ index ]

    def ReturnOCellSynapses( self ):
    # Return the OCell synapses as ( OCell, Permanence ).

        oCellList = []

        for index, oCell in enumerate( self.OCellSynapses ):
            oCellList.append( ( oCell, self.segmentConfidence ) )
#            oCellList.append( ( oCell, self.OCellPermanences[ index ] * self.stimulated ) )

        return oCellList

# BELOW CAN BE DELETED
    def AdjustVectorProperties( self ):
    # Adjusts segments vector score properties based on lastProbabilityScore, if this segment accurately predicted the winner terminal cell.

#        if self.OCellOVerlap:
        return False
            # If last vector was one standardDeviation away from center.


        # If 0.0 < probabilityScore < vectorLowThresh... then widen standardDeviation and lower vectorConfidence.
        if self.lastProbabilityScore > 0.0 and self.lastProbabilityScore < self.vectorMemoryDict[ "vectorScoreLowerThreshold" ]:
            self.standardDeviation -= self.vectorMemoryDict[ "vectorScaleShift" ]
            if self.standardDeviation < 0.0001:
                self.standardDeviation = 0.0001
            self.vectorConfidence -= self.vectorMemoryDict[ "vectorConfidenceShift" ]
            if self.vectorConfidence < 0.0001:
                self.vectorConfidence = 0.0001

        # If vectorLowThresh <= probabilityScore < vectorUpThresh... then shift vectorCenter.
#        elif self.lastProbabilityScore >= self.vectorMemoryDict[ "vectorScoreLowerThreshold" ] and self.lastProbabilityScore < self.vectorMemoryDict[ "vectorScoreUpperThreshold" ]:
#            for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):

        # If vectorUpThresh <= probabilityScore <= 1.0... then tighten standardDeviation and raise vectorConfidence.
        elif self.lastProbabilityScore <= self.vectorMemoryDict[ "vectorScoreUpperThreshold" ] and self.lastProbabilityScore <= 1.0:
            self.standardDeviation += self.vectorMemoryDict[ "vectorScaleShift" ]
            if self.standardDeviation > 0.01:
                self.standardDeviation = 0.01
            self.vectorConfidence += self.vectorMemoryDict[ "vectorConfidenceShift" ]
            if self.vectorConfidence > 1.0:
                self.vectorConfidence = 1.0

    def ChangeVectorStrength( self, vector, activeOCells ):
    # Modifies declared vector position by permanenceAdjust, and smooths this out.

        if self.vectorMemoryDict[ "vectorDimensions" ] != 2:
            print( "ChangeVectorStrength only works for 2 dimensions." )
            exit()

        OCellOVerlap = FastIntersect( activeOCells, self.OCellSynapses )
        if len( OCellOVerlap ) >= self.vectorMemoryDict[ "objectRepActivation" ]:
            oCellsAgree = True
        else:
            oCellsAgree = False

        vectorIndices = []
        for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
            vectorIndices.append( int( ( vector[ d ] + ( self.vectorMemoryDict[ "vectorRange" ] * 0.5 ) ) * self.vectorMemoryDict[ "numVectorSynapses" ] / self.vectorMemoryDict[ "vectorRange" ] ) )

        for x in range( -self.vectorMemoryDict[ "maxVectorSynapseRadius" ], self.vectorMemoryDict[ "maxVectorSynapseRadius" ] ):
            for y in range( -self.vectorMemoryDict[ "maxVectorSynapseRadius" ], self.vectorMemoryDict[ "maxVectorSynapseRadius" ] ):
                indX = x + vectorIndices[ 0 ]
                indY = y + vectorIndices[ 1 ]
                if indX >= 0 and indX < self.vectorMemoryDict[ "numVectorSynapses" ] and indY >= 0 and indY < self.vectorMemoryDict[ "numVectorSynapses" ]:
                        distanceSquared = ( x * x ) + ( y * y )
                        if distanceSquared <= self.vectorMemoryDict[ "maxVectorSynapseRadius" ] ** 2:
                            synapseModifier = self.vectorMemoryDict[ "vectorSynapseScaleFactor" ] ** math.sqrt( distanceSquared )
                            if oCellsAgree:
                                self.vectorSynapses[ indX ][ indY ] = ModThisSynapse( self.vectorSynapses[ indX ][ indY ], self.vectorMemoryDict[ "permanenceIncrement" ], 1.0, 0.0, True )
                            else:
                                self.vectorSynapses[ indX ][ indY ] = ModThisSynapse( self.vectorSynapses[ indX ][ indY ], -self.vectorMemoryDict[ "permanenceDecrement" ], 1.0, 0.0, True )

    def Inside( self, vector ):
    # Checks if given vector position is inside range. Calculates this based on a score between 0.0 and 1.0, which is used as a probability.

        # Calculate vector probability.
#        self.lastProbabilityScore = CalculateDistanceScore( vector, self.vectorMemoryDict[ "vectorDimensions" ], self.vectorCenter, self.standardDeviation, self.vectorConfidence )

#        for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
#            self.lastVector[ d ] = vector[ d ]

#        if self.lastProbabilityScore >= self.vectorMemoryDict[ "vectorScoreLowerThreshold" ]:

        vectorIndices = []
        for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
            vectorIndices.append( int( ( vector[ d ] + ( self.vectorMemoryDict[ "vectorRange" ] * 0.5 ) ) * self.vectorMemoryDict[ "numVectorSynapses" ] / self.vectorMemoryDict[ "vectorRange" ] ) )

        if self.vectorSynapses[ vectorIndices[ 0 ] ][ vectorIndices[ 1 ] ] >= self.vectorMemoryDict[ "vectorScoreLowerThreshold" ]:
            self.lastProbabilityScore = self.vectorSynapses[ vectorIndices[ 0 ] ][ vectorIndices[ 1 ] ]
            return True, self.lastProbabilityScore
        else:
            return False, self.lastProbabilityScore

    def IncidentColActive( self, incidentCol ):
    # Takes the incident cell and adds it to incidentActivation.

        NoRepeatInsort( self.incidentActivation, incidentCol )

    def CheckOCellOverlap( self, activeOCells, OCellThreshold ):
    # Check the overlap of this segments OCells against the active OCells. Return this terminal cell if above threshold.

        overlap = len( FastIntersect( self.OCellSynapses, activeOCells ) )

        if overlap >= OCellThreshold:
            self.OCellOVerlap = True
            return self.terminalSynapse
        else:
            self.OCellOVerlap = False
            return None

    def UpdateStimulation( self, activationScore ):
    # If this segment is stimulated ( self.stimulated > 0.0 ) then update its stimulation given timeSinceActive and prediction score.

        if activationScore != None:
            self.stimulated = ( -1 * ( 2 ** ( -0.1 * activationScore ) ) ) + 1

        if self.stimulated > 0.0:
            self.stimulated -= self.timeSinceActive * 0.1

        if self.stimulated <= 0.0:
            self.stimulated      = 0.0
            return False
        else:
            return True

    def CheckActivation( self, vector, activeOCells ):
    # Checks the incidentActivation against activationThreshold to see if segment becomes active.

        if len( self.incidentActivation ) >= self.activationThreshold:
            self.activeAboveThresh = True

            inside, vectorScore = self.Inside( vector )

            if inside:
                self.active            = True
                self.segmentConfidence = len( self.incidentActivation ) * vectorScore
                self.timeSinceActive   = 0

                self.UpdateStimulation( len( self.incidentActivation ) * vectorScore )

                return True

        self.active = False
        return False

    def RefreshSegment( self, maxTimeSinceActive, resetStimulated ):
    # Updates or refreshes the state of the segment.

        self.active             = False
        self.incidentActivation = []
        self.timeSinceActive   += 1
        self.activeAboveThresh  = False
        self.OCellOVerlap       = False

        if resetStimulated:
            self.stimulated = 0.0
            stillStimulated = False
        else:
            stillStimulated = self.UpdateStimulation( None )

        if self.timeSinceActive > maxTimeSinceActive:
            return True, stillStimulated
        else:
            return False, stillStimulated

    def RemoveIncidentSynapse( self, columnToDelete, cellsPerColumn ):
    # Delete the incident synapse sent.

        index = IndexIfItsIn( self.incidentColumns, columnToDelete )
        if index != None:
            del self.incidentColumns[ index ]
        else:
            print( "Attempt to remove synapse from segment, but synapse doesn't exist." )
            exit()

        if int( self.incidentSynapses[ index ] / cellsPerColumn ) == columnToDelete:
            del self.incidentSynapses[ index ]
        else:
            print( "Deleted column doesn't match cell." )
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

    def AdjustThreshold( self ):
    # Use the segments vectorConfidence to adjust activationThreshold.

        self.activationThreshold = ( ( self.vectorMemoryDict[ "FActivationThresholdMax" ] - self.vectorMemoryDict[ "FActivationThresholdMin" ] ) * 2 ** ( 3 * ( self.vectorConfidence - 1 ) ) ) + self.vectorMemoryDict[ "FActivationThresholdMin" ]

    def CellForColumn( self, column ):
    # Return the cell for the column.

        index = IndexIfItsIn( self.incidentColumns, column )
        if index != None:
            return self.incidentSynapses[ index ]
        else:
            return None

#-------------------------------------------------------------------------------

class SegmentStructure:

    def __init__( self, vectorMemoryDict ):
    # Initialize the segment storage and handling class.

        self.vectorMemoryDict = vectorMemoryDict

        self.segments           = []                                  # Stores all segments structures.
        self.activeSegments     = []
        self.stimulatedSegments = []

        self.segsToDelete   = []

        self.incidentSegments = []                          # Stores the connections from each incident cell to segment.
        for cell in range( vectorMemoryDict[ "cellsPerColumn" ] * vectorMemoryDict[ "columnDimensions" ] ):
            self.incidentSegments.append( [] )
        self.incidentPermanences = []
        for col in range(  vectorMemoryDict[ "columnDimensions" ] ):
            self.incidentPermanences.append( [] )

        # Working Memory portion.-----------------------------------------------
        self.numPositionCells   = 100
        self.maxPositionRange   = 800

    def HowManyActiveSegs( self ):
    # Return the number of active segments.

        return len( self.activeSegments )

    def AddSynapse( self, incCell, segIndex ):
    # Add a synapse to specified segment.

        incCol = int( incCell / self.vectorMemoryDict[ "cellsPerColumn" ] )

        insertIndex = bisect_left( self.incidentSegments[ incCol ], segIndex )

        if insertIndex != len( self.incidentSegments[ incCol ] ) and self.incidentSegments[ incCol ][ insertIndex ] == segIndex:
            print( "Synapse to this segment already exists." )
            exit()

        self.incidentSegments[ incCol ].insert( insertIndex, segIndex )
        self.incidentPermanences[ incCol ].insert( insertIndex, self.vectorMemoryDict[ "initialPermanence" ] )
        self.segments[ segIndex ].NewIncidentSynapse( incCell, incCol )

    def DeleteSynapse( self, incCol, segIndex ):
    # Remove a synapse to specified segment.

        # Check if synapse exists and if so delete it and all references to it.
        delIndex = bisect_left( self.incidentSegments[ incCol ], segIndex )

        if delIndex != len( self.incidentSegments[ incCol ] ) and self.incidentSegments[ incCol ][ delIndex ] == segIndex:
            del self.incidentSegments[ incCol ][ delIndex ]
            del self.incidentPermanences[ incCol ][ delIndex ]
            self.segments[ segIndex ].RemoveIncidentSynapse( incCol, self.vectorMemoryDict[ "cellsPerColumn" ] )

        # Check if segment has any synapses left. If none then mark segment for deletion.
        if len( self.segments[ segIndex ].incidentSynapses ) == 0:
            self.segsToDelete.append( segIndex )

        return delIndex

    def DeleteSegmentsAndSynapse( self, Cells ):
    # Receives a list of indices of segments that need deletion. Deletes these segments from self.segments,
    # and removes all references to them in self.incidentSegments.

        if len( self.segsToDelete ) > 0:
            self.segsToDelete.sort()

            for segIndex in reversed( self.segsToDelete ):
                # Delete any references to segment, and lower the index of all greater segment reference indices by one.
                for incCol, incList in enumerate( self.incidentSegments ):
                    indexAt = self.DeleteSynapse( incCol, segIndex )

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

                staIndex = bisect_left( self.stimulatedSegments, segIndex )
                if staIndex != len( self.stimulatedSegments ) and self.stimulatedSegments[ staIndex ] == segIndex:
                    del self.stimulatedSegments[ staIndex ]
                while staIndex < len( self.stimulatedSegments ):
                    self.stimulatedSegments[ staIndex ] -= 1
                    staIndex += 1

                # Delete the segment.
                del self.segments[ segIndex ]

        self.segsToDelete = []

    def CreateSegment( self, Cells, incidentCellsList, terminalCell, vector, objectCellsList ):
    # Creates a new segment.

        newSegment = Segment( self.vectorMemoryDict, terminalCell, vector, objectCellsList )
        self.segments.append( newSegment )

        indexOfNew = len( self.segments ) - 1
        self.activeSegments.append( indexOfNew )

        for incCell in incidentCellsList:
            self.AddSynapse( incCell, indexOfNew )

        if terminalCell != None:
            Cells[ terminalCell ].isTerminalCell += 1

    def UpdateSegmentActivity( self, FCells ):
    # Make every segment that was active inactive, and refreshes its synapse activation.
    # Also add a time step to each segment, and see if it dies as a result. Delete any segments that die.

        self.activeSegments = []

        # Increase time for all segments and deactivate them and alter their state.
        for index, segment in enumerate( self.segments ):
            deadTooLong, stillStimulated = segment.RefreshSegment( self.vectorMemoryDict[ "segmentDecay" ], False )

            if deadTooLong:
                self.segsToDelete.append( index )

            if not stillStimulated:
                i = IndexIfItsIn( self.stimulatedSegments, index )
                if i != None:
                    del self.stimulatedSegments[ i ]

        self.DeleteSegmentsAndSynapse( FCells )

    def OCellSegmentLearning( self, activeOCells, OCells ):
    # For the active OCells enforce this on all stimulated segments, causing them to form synapses to these if they don't have them,
    # and supporting synapses if they exist.

        for actSeg in self.activeSegments:
            self.segments[ actSeg ].OCellLearning( OCells, activeOCells, self.vectorMemoryDict[ "initialPermanence" ], self.vectorMemoryDict[ "permanenceIncrement" ],
                self.vectorMemoryDict[ "permanenceDecrement" ], self.vectorMemoryDict[ "maxSynapsesToAddPer" ], self.vectorMemoryDict[ "maxSynapsesPerSegment" ] )

    def AdjustThresholds( self, lastVector, activeOCells ):
    # Adjust the SDR threshold and vector score properties for active segments.

        for activeSeg in self.activeSegments:
            # Adjust the segments vector properties. If the segment agrees with the active OCells then support the vector position, if not then decrease.
#            self.segments[ activeSeg ].AdjustVectorProperties()

            self.segments[ activeSeg ].ChangeVectorStrength( lastVector, activeOCells )

        # self.segments[ activeSeg ].AdjustThreshold()


    def SegmentLearning( self, FCells, OCells, lastWinnerCells, lastActiveCells, activeOCells, lastVector ):
    # Perform learning on all active and inactive segments.
    # Refresh all segments then perform learning on them.
    # Delete segments that need deleting.

        self.DecayAndCreate( FCells, lastWinnerCells, lastActiveCells )

        self.AdjustThresholds( lastVector, activeOCells )

        self.OCellSegmentLearning( activeOCells, OCells )

        self.CheckIfSegsIdentical( FCells )

    def GetStimulatedOCells( self, numberOCells ):
    # Return the OCells count from all stimulated segments by summing their permances.

        OCellCounts = [ 0.0 ] * numberOCells

        for stimSeg in self.stimulatedSegments:
            for oCell in self.segments[ stimSeg ].ReturnOCellSynapses():
                OCellCounts[ oCell[ 0 ] ] += oCell[ 1 ]

        return OCellCounts

    def ResetStimulatedSegments( self ):
    # Turn off all stimulated segments.

        for stimSeg in self.stimulatedSegments:
            self.segments[ stimSeg ].RefreshSegment( self.vectorMemoryDict[ "segmentDecay" ], True )

    def StimulateSegments( self, activeOCells, activeColumns, vector ):
    # Using the activeCells and vector find all segments that activate. Add these segments to a list and return it.

        predictedCells = []

        if len( self.incidentSegments ) > 0:
            # Activate the synapses in segments using activeCells.
            for incCol in activeColumns:
                for entry in self.incidentSegments[ incCol ]:
                    self.segments[ entry ].IncidentColActive( incCol )

            # Check the overlap of all segments and see which ones are active, and add the terminalCell to stimulatedCells.
            for segIndex, segment in enumerate( self.segments ):
                if segment.CheckActivation( vector, activeOCells ):
                    NoRepeatInsort( self.activeSegments, segIndex )
                    NoRepeatInsort( self.stimulatedSegments, segIndex )

                    terminalCell = segment.CheckOCellOverlap( activeOCells, 20 )
                    if terminalCell != None:
                        # Terminal cell becomes predictive.
                        NoRepeatInsort( predictedCells, terminalCell )

        return predictedCells

    def GetSegmentAverages( self, Cells ):
    # Gets the active segments which terminate on a winner cell and gathers their vector data for working memory.

        averages = [ 0, [ 0.0 ] * self.vectorMemoryDict[ "vectorDimensions" ], [ 0.0 ] * self.vectorMemoryDict[ "vectorDimensions" ], 0 ]          # [ Segment count, Sum of standardDeviation, Sum of vectorConfidence, Sum of activation thresholds ]

        for activeSeg in self.activeSegments:
            if Cells[ self.segments[ activeSeg ].terminalSynapse ].winner:
                averages[ 0 ] += 1
                for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
                    averages[ 1 ][ d ] += self.segments[ activeSeg ].standardDeviation[ d ]
                    averages[ 2 ][ d ] += self.segments[ activeSeg ].vectorConfidence[ d ]
                averages[ 3 ] += self.segments[ activeSeg ].activationThreshold

        return averages

    def ChangePermanence( self, incCell, segIndex, permanenceChange ):
    # Change the permanence of synapse incident on incCell, part of segIndex, by permanenceChange.
    # If permanence == 0.0 then delete it.

        incCol = int( incCell / self.vectorMemoryDict[ "cellsPerColumn" ] )

        entryIndex = IndexIfItsIn( self.incidentSegments[ incCol ], segIndex )
        if entryIndex != None:
            self.incidentPermanences[ incCol ][ entryIndex ] = ModThisSynapse( self.incidentPermanences[ incCol ][ entryIndex ], permanenceChange, 1.0, 0.0, True )

            if self.incidentPermanences[ incCol ][ entryIndex ] <= 0.0:
                self.DeleteSynapse( incCol, segIndex )

    def DecayAndCreate( self, Cells, lastWinnerCells, lastActiveCells ):
    # For all active segments:
    # 1.) Decrease all synapses on active segments where the terminal cell is not a winner.
    # 2.) If terminal cell is winner increase condfidence for segment;
    # 3.) Increase synapse strength to active incident cells that already have synapses;
    # 4.) Decrease synapse strength to inactive incident cells that already have synapses;
    # 5.) Build new synapses to active incident winner cells that don't have synapses;

        # Then deal with active segments.
        for activeSeg in self.activeSegments:
            # 1.)...
            if self.segments[ activeSeg ].terminalSynapse != None and not Cells[ self.segments[ activeSeg ].terminalSynapse ].winner:

                for incCell in self.segments[ activeSeg ].incidentSynapses:
                    self.ChangePermanence( incCell, activeSeg, -self.vectorMemoryDict[ "permanenceDecrement" ] )
            # 2.)...
            else:
                synapseToAdd = lastWinnerCells.copy()
                for incCell in self.segments[ activeSeg ].incidentSynapses:
                    # 3.)...
                    if BinarySearch( lastActiveCells, incCell ):
                        self.ChangePermanence( incCell, activeSeg, self.vectorMemoryDict[ "permanenceIncrement" ] )
                    # 4.)...
                    else:
                        self.ChangePermanence( incCell, activeSeg, -self.vectorMemoryDict[ "permanenceDecrement" ] )

                    indexIfIn = IndexIfItsIn( synapseToAdd, incCell )
                    if indexIfIn != None:
                        del synapseToAdd[ indexIfIn ]

                if len( synapseToAdd ) > 0:
                    # 5.)...
                    # Check to make sure this segment doesn't already have a synapse to this column.
                    realSynapsesToAdd = []
                    for synAdd in synapseToAdd:
                        if not self.segments[ activeSeg ].AlreadySynapseToColumn( synAdd, self.vectorMemoryDict[ "cellsPerColumn" ] ):
                            realSynapsesToAdd.append( synAdd )

                    reallyRealSynapsesToAdd = sample( realSynapsesToAdd, min( len( realSynapsesToAdd ), self.vectorMemoryDict[ "maxSynapsesToAddPer" ] ) )
                    for toAdd in reallyRealSynapsesToAdd:
                        self.AddSynapse( toAdd, activeSeg )

                    # If the number of synapses is above maxSynapsesPerSegment then delete random synapses.
                    while self.vectorMemoryDict[ "maxSynapsesPerSegment" ] - len( self.segments[ activeSeg ].incidentSynapses ) < 0:
                        toDel = choice( self.segments[ activeSeg ].incidentSynapses )
                        self.ChangePermanence( toDel, activeSeg, -1.0 )

    def ThereCanBeOnlyOne( self, cellsList ):
    # Return the cell, in cellsList, which is terminal on an active segment with greatest activation.

        greatestActivation = 0.0
        greatestCell       = cellsList[ 0 ]

        if len( cellsList ) > 1:
            for cell in cellsList:
                for actSeg in self.activeSegments:
                    if self.segments[ actSeg ].terminalSynapse == cell:
                        thisActivation = self.segments[ actSeg ].lastProbabilityScore * len( self.segments[ actSeg ].incidentActivation )
#                        thisActivation = self.ReturnActivation( actSeg )
                        if thisActivation > greatestActivation:
                            greatestActivation = thisActivation
                            greatestCell       = cell

        return ( greatestCell, greatestActivation )

    def ReturnActivation( self, seg ):
    # Returns the sum of permanences of synapses incident on incCellList and part of seg.

        activation = 0.0

        for incCol in self.segments[ seg ].incidentColumns:
            entryIndex = IndexIfItsIn( self.incidentSegments[ incCol ], seg )
            if entryIndex != None:
                activation += self.incidentPermanences[ incCol ][ entryIndex ]

        return activation

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
                        if not segment.Equality( self.segments[ entrySegment ], self.vectorMemoryDict[ "equalityThreshold" ] ):
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
                winnerSegment = group.pop( randrange( len( group ) ) )

                for segIndex in range( len( group ) ):
                    print("DELETED IDENTIAL SEGMENTS--------------------")
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

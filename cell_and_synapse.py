from bisect import bisect_left
from random import uniform, choice, sample, randrange, shuffle
from collections import Counter
from useful_functions import BinarySearch, NoRepeatInsort, ModThisSynapse, IndexIfItsIn, FastIntersect, GenerateUnitySDR, NumStandardDeviations, CalculateDistanceScore
import numpy
import math
#from time import time

class Segment:

    def __init__( self, vectorMemoryDict, incidentColumns, terminalColumn, vector ):
    # Initialize the inidividual segment structure class.

        self.vectorMemoryDict = vectorMemoryDict

        self.active              = False                         # True means it's predictive, and above threshold terminal cells fired.
        self.stimulated          = 1.0
        self.timeSinceActive     = 0                            # Time steps since this segment was active last.
        self.activationThreshold = vectorMemoryDict[ "FActivationThresholdMin" ]   # Minimum overlap required to activate segment.
        self.activeAboveThresh   = False

        # Lateral synapse portion.
        self.incidentSynapses    = []
        self.incidentPermanences = []
        for iCol in incidentColumns:
            for iCell in range( iCol * vectorMemoryDict[ "cellsPerColumn" ], ( iCol * vectorMemoryDict[ "cellsPerColumn" ] ) + vectorMemoryDict[ "cellsPerColumn" ] ):
                self.incidentSynapses.append( iCell )
                self.incidentPermanences.append( uniform( 0, 1 ) )
        self.incidentColumns     = incidentColumns.copy()
        self.terminalSynapses    = []           # Generate random synapse connections to all cells in terminal column.
        self.terminalColumn      = terminalColumn
        self.terminalPermanences = []
        for tCell in range( terminalColumn * vectorMemoryDict[ "cellsPerColumn" ], ( terminalColumn * vectorMemoryDict[ "cellsPerColumn" ] ) + vectorMemoryDict[ "cellsPerColumn" ] ):
            self.terminalSynapses.append( tCell )
            self.terminalPermanences.append( uniform( 0, 1 ) )

        self.incidentActivation  = []           # A list of all columns that last overlapped with incidentSynapses.

        # Vector portion.
        self.vectorCenter        = vector                      # The origin of the vector.
        self.relativeVector    = [ 0 ] * len( vector )

#        self.standardDeviation   = [ self.vectorMemoryDict[ "initialStandardDeviation" ] ] * self.vectorMemoryDict[ "vectorDimensions" ]
#        self.vectorConfidence    = [ self.vectorMemoryDict[ "initialVectorConfidence" ] ] * self.vectorMemoryDict[ "vectorDimensions" ]

#        self.lastProbabilityScore = 0
#        self.segmentConfidence    = 0.0
#        self.lastVector = [ 0 ] * self.vectorMemoryDict[ "vectorDimensions" ]

#        self.vectorRange       = []
#        for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
#            self.vectorRange.append( [ -self.vectorMemoryDict[ "vectorRange" ], self.vectorMemoryDict[ "vectorRange" ] ] )

#        if vector != None:
#            if len( vector ) != self.vectorMemoryDict[ "vectorDimensions" ]:
#                print( "Vector sent to create segment not of same dimensions sent." )
#                exit()

#            self.vectorSynapses = numpy.zeros( shape = [ self.vectorMemoryDict[ "numVectorSynapses" ] ] * self.vectorMemoryDict[ "vectorDimensions" ] )
# MIGHT NEED TO ADD A SCALE FUNCTION, AND RANGE FUNCTION INTO THIS LATER.
#            for y in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
#                self.ChangeVectorStrength( y, vector[ y ], self.vectorMemoryDict[ "initialPermanence" ] )
#        else:
#            self.vectorSynapses = vectorCells

#        # Object synapse portion.
#        self.OCellSynapses    = objectCellsList.copy()
#        self.OCellPermanences = [ self.vectorMemoryDict[ "initialPermanence" ] ] * len( objectCellsList )
#        self.OCellOVerlap     = False

#        if len( vector ) != self.vectorMemoryDict[ "vectorDimensions" ]:
#            print( "Vector sent to create segment not of same dimensions sent." )
#            exit()

#    def OCellLearning( self, OCells, activeOCells, initialPermanence, permanenceIncrement, permanenceDecrement, maxToAdd, maxSynapses ):
#    # Support synapses to activeOCells, and create new if don't exist. Decay synapses to non-activeOCells.
#
#        synapseToAdd    = activeOCells.copy()
#        synapseToDelete = []
#
#        for synIndex, syn in enumerate( self.OCellSynapses ):
#            if not OCells[ syn ].active:
#                self.OCellPermanences[ synIndex ] = ModThisSynapse( self.OCellPermanences[ synIndex ], -permanenceDecrement, 1.0, 0.0, True )
#                if self.OCellPermanences[ synIndex ] <= 0.0:
#                    synapseToDelete.append( synIndex )
#            else:
#                self.OCellPermanences[ synIndex ] = ModThisSynapse( self.OCellPermanences[ synIndex ], permanenceIncrement, 1.0, 0.0, True )
#                i = IndexIfItsIn( synapseToAdd, syn )
#                if i != None:
#                    del synapseToAdd[ i ]
#                else:
#                    print( "Active OCell missing from activeOCells in OCellLearning()" )
#                    exit()
#
#        if len( synapseToDelete ) > 0:
#            for toDel in reversed( synapseToDelete ):
#                del self.OCellSynapses[ toDel ]
#                del self.OCellPermanences[ toDel ]
#
#        if len( synapseToAdd ) > 0:
#            numAdded = 0
#            while len( self.OCellSynapses ) <= maxSynapses and numAdded <= maxToAdd and len( synapseToAdd ) > 0:
#                index = randrange( len( synapseToAdd ) )
#                insertion = bisect_left( self.OCellSynapses, synapseToAdd[ index ] )
#                if insertion == len( self.OCellSynapses ):
#                    self.OCellSynapses.append( synapseToAdd[ index ] )
#                    self.OCellPermanences.append( initialPermanence )
#                else:
#                    self.OCellSynapses.insert( insertion, synapseToAdd[ index ] )
#                    self.OCellPermanences.insert( insertion, initialPermanence )
#
#                numAdded += 1
#                del synapseToAdd[ index ]

#    def ReturnOCellSynapses( self ):
#    # Return the OCell synapses as ( OCell, Permanence ).
#
#        oCellList = []
#
#        for index, oCell in enumerate( self.OCellSynapses ):
#            oCellList.append( ( oCell, self.segmentConfidence ) )
#            oCellList.append( ( oCell, self.OCellPermanences[ index ] * self.stimulated ) )
#
#        return oCellList

# BELOW CAN BE DELETED
#    def AdjustVectorProperties( self ):
#    # Adjusts segments vector score properties based on lastProbabilityScore, if this segment accurately predicted the winner terminal cell.
#
#        if self.OCellOVerlap:
#        return False
#            # If last vector was one standardDeviation away from center.
#
#
#        # If 0.0 < probabilityScore < vectorLowThresh... then widen standardDeviation and lower vectorConfidence.
#        if self.lastProbabilityScore > 0.0 and self.lastProbabilityScore < self.vectorMemoryDict[ "vectorScoreLowerThreshold" ]:
#            self.standardDeviation -= self.vectorMemoryDict[ "vectorScaleShift" ]
#            if self.standardDeviation < 0.0001:
#                self.standardDeviation = 0.0001
#            self.vectorConfidence -= self.vectorMemoryDict[ "vectorConfidenceShift" ]
#            if self.vectorConfidence < 0.0001:
#                self.vectorConfidence = 0.0001
#
#        # If vectorLowThresh <= probabilityScore < vectorUpThresh... then shift vectorCenter.
#        elif self.lastProbabilityScore >= self.vectorMemoryDict[ "vectorScoreLowerThreshold" ] and self.lastProbabilityScore < self.vectorMemoryDict[ "vectorScoreUpperThreshold" ]:
#            for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
#
#        # If vectorUpThresh <= probabilityScore <= 1.0... then tighten standardDeviation and raise vectorConfidence.
#        elif self.lastProbabilityScore <= self.vectorMemoryDict[ "vectorScoreUpperThreshold" ] and self.lastProbabilityScore <= 1.0:
#            self.standardDeviation += self.vectorMemoryDict[ "vectorScaleShift" ]
#            if self.standardDeviation > 0.01:
#                self.standardDeviation = 0.01
#            self.vectorConfidence += self.vectorMemoryDict[ "vectorConfidenceShift" ]
#            if self.vectorConfidence > 1.0:
#                self.vectorConfidence = 1.0

#    def ChangeVectorStrength( self, vector, activeOCells ):
#    # Modifies declared vector position by permanenceAdjust, and smooths this out.
#
#        if self.vectorMemoryDict[ "vectorDimensions" ] != 2:
#            print( "ChangeVectorStrength only works for 2 dimensions." )
#            exit()
#
#        OCellOVerlap = FastIntersect( activeOCells, self.OCellSynapses )
#        if len( OCellOVerlap ) >= self.vectorMemoryDict[ "objectRepActivation" ]:
#            oCellsAgree = True
#        else:
#            oCellsAgree = False
#
#        vectorIndices = []
#        for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
#            vectorIndices.append( int( ( vector[ d ] + ( self.vectorMemoryDict[ "vectorRange" ] * 0.5 ) ) * self.vectorMemoryDict[ "numVectorSynapses" ] / self.vectorMemoryDict[ "vectorRange" ] ) )
#
#        for x in range( -self.vectorMemoryDict[ "maxVectorSynapseRadius" ], self.vectorMemoryDict[ "maxVectorSynapseRadius" ] ):
#            for y in range( -self.vectorMemoryDict[ "maxVectorSynapseRadius" ], self.vectorMemoryDict[ "maxVectorSynapseRadius" ] ):
#                indX = x + vectorIndices[ 0 ]
#                indY = y + vectorIndices[ 1 ]
#                if indX >= 0 and indX < self.vectorMemoryDict[ "numVectorSynapses" ] and indY >= 0 and indY < self.vectorMemoryDict[ "numVectorSynapses" ]:
#                        distanceSquared = ( x * x ) + ( y * y )
#                        if distanceSquared <= self.vectorMemoryDict[ "maxVectorSynapseRadius" ] ** 2:
#                            synapseModifier = self.vectorMemoryDict[ "vectorSynapseScaleFactor" ] ** math.sqrt( distanceSquared )
#                            if oCellsAgree:
#                                self.vectorSynapses[ indX ][ indY ] = ModThisSynapse( self.vectorSynapses[ indX ][ indY ], self.vectorMemoryDict[ "permanenceIncrement" ], 1.0, 0.0, True )
#                            else:
#                                self.vectorSynapses[ indX ][ indY ] = ModThisSynapse( self.vectorSynapses[ indX ][ indY ], -self.vectorMemoryDict[ "permanenceDecrement" ], 1.0, 0.0, True )

    def Inside( self, vector, vectorCheck ):
    # Checks if given vector position is inside range. Calculates this based on a score between 0.0 and 1.0, which is used as a probability.

        # Calculate vector probability that we are inside.
        numStdDev = NumStandardDeviations( vector, self.vectorMemoryDict[ "vectorDimensions" ], vectorCheck, self.vectorMemoryDict[ "initialStandardDeviation" ] )

        if numStdDev <= self.vectorMemoryDict[ "initialVectorConfidence" ]:
            return True
        else:
            return False

#        self.lastProbabilityScore = CalculateDistanceScore( vector, self.vectorMemoryDict[ "vectorDimensions" ], self.vectorCenter, self.standardDeviation, self.vectorConfidence )

#        for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
#            self.lastVector[ d ] = vector[ d ]

#        if self.lastProbabilityScore >= self.vectorMemoryDict[ "vectorScoreLowerThreshold" ]:

#        vectorIndices = []
#        for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
#            vectorIndices.append( int( ( vector[ d ] + ( self.vectorMemoryDict[ "vectorRange" ] * 0.5 ) ) * self.vectorMemoryDict[ "numVectorSynapses" ] / self.vectorMemoryDict[ "vectorRange" ] ) )
#
#        print(self.vectorSynapses[ vectorIndices[ 0 ] ][ vectorIndices[ 1 ] ])
#
#        if self.vectorSynapses[ vectorIndices[ 0 ] ][ vectorIndices[ 1 ] ] >= self.vectorMemoryDict[ "vectorScoreLowerThreshold" ]:
#            self.lastProbabilityScore = self.vectorSynapses[ vectorIndices[ 0 ] ][ vectorIndices[ 1 ] ]
#            return True, self.lastProbabilityScore
#        else:
#            return False, self.lastProbabilityScore

    def IncidentCellActive( self, incidentCell, incidentColumn ):
    # Takes the incident cell and adds it to incidentActivation, if above permanenceLowerThreshold.

        index = IndexIfItsIn( self.incidentSynapses, incidentCell )
        if index != None:
            if self.incidentPermanences[ index ] >= self.vectorMemoryDict[ "permanenceLowerThreshold" ]:
                NoRepeatInsort( self.incidentActivation, incidentColumn )

    def UpdateRelativeVector( self, vector ):
    # Uses vector to update relativeVector.

        for d in range( len( vector ) ):
            self.relativeVector[ d ] += vector[ d ]

#    def CheckOCellOverlap( self, activeOCells, OCellThreshold ):
#    # Check the overlap of this segments OCells against the active OCells. Return this terminal cell if above threshold.
#
#        overlap = len( FastIntersect( self.OCellSynapses, activeOCells ) )
#
#        if overlap >= OCellThreshold:
#            self.OCellOVerlap = True
#            return self.terminalSynapse
#        else:
#            self.OCellOVerlap = False
#            return None

#    def UpdateStimulation( self, activationScore ):
#    # If this segment is stimulated ( self.stimulated > 0.0 ) then update its stimulation given timeSinceActive and prediction score.
#
#        if activationScore != None:
#            self.stimulated = ( -1 * ( 2 ** ( -0.1 * activationScore ) ) ) + 1
#
#        if self.stimulated > 0.0:
#            self.stimulated -= self.timeSinceActive * 0.1
#
#        if self.stimulated <= 0.0:
#            self.stimulated      = 0.0
#            return False
#        else:
#            return True

    def IncreaseTerminalPermanence( self, terIndex ):
    # Increases the permanence of indexed terminal cell.

        self.terminalPermanences[ terIndex ] += self.vectorMemoryDict[ "permanenceIncrement" ]

        if self.terminalPermanences[ terIndex ] >= 1.0:
            self.terminalPermanences[ terIndex ] = 1.0

    def DecreaseTerminalPermanence( self, terIndex ):
    # Decreases permanence of indexed terminal cell.

        self.terminalPermanences[ terIndex ] -= self.vectorMemoryDict[ "permanenceDecrement" ]

        if self.terminalPermanences[ terIndex ] <= 0.0:
            # If it is below 0.0 then delete it.
            del self.terminalSynapses[ terIndex ]
            del self.terminalPermanences[ terIndex ]

# IF NO TERMINAL SYNAPSES THEN MARK FOR DELETION. CAN DELETE SEGMENTS BY MARKING THEM FOR DELETION AND THEN JUST GOING THROUGH EACH ONE AND CHECKING.

    def IncreaseIncidentPermanence( self, incIndex ):
    # Increase permanence to indexed incident cell.

        self.incidentPermanences[ incIndex ] += self.vectorMemoryDict[ "permanenceIncrement" ]

        if self.incidentPermanences[ incIndex ] >= 1.0:
            self.incidentPermanences[ incIndex ] = 1.0

    def DecreaseIncidentPermanence( self, incIndex ):
    # Decreases permanence of indexed incident cell.

        self.incidentPermanences[ incIndex ] -= self.vectorMemoryDict[ "permanenceDecrement" ]

        if self.incidentPermanences[ incIndex ] <= 0.0:
            # If it is below 0.0 then delete it.
            del self.incidentSynapses[ incIndex ]
            del self.incidentPermanences[ incIndex ]

# IF INCIDENT SYNAPSES BELOW SOME LEVEL THEN MARK FOR DELETION. CAN DELETE SEGMENTS BY MARKING THEM FOR DELETION AND THEN JUST GOING THROUGH EACH ONE AND CHECKING.

    def ModifyAllSynapses( self, FCells ):
    # 1.) Decrease all terminal permanences to currently non-winner cells.
    # 2.) If terminal cell is winner increase permance for this terminal synapse;
    # 3.) Increase permanence strength to last-active incident cells that already have synapses;
    # 4.) Decrease synapse strength to inactive incident cells that already have synapses;
    # 5.) Build new synapses to active incident winner cells that don't have synapses;

            # 1.)...
            for terIndex, terCell in enumerate( self.terminalSynapses ):
                if not FCells[ terCell ].winner:
                    self.DecreaseTerminalPermanence( terIndex )
            # 2.)...
                else:
                    self.IncreaseTerminalPermanence( terIndex )

#            synapseToAdd = lastActiveFCells.copy()                          # Used for 5 above.

            for incIndex, incCell in enumerate( self.incidentSynapses ):
                # 3.)...
                if FCells[ incCell ].lastActive:
                    self.IncreaseIncidentPermanence( incIndex )
                # 4.)...
                else:
                    self.DecreaseIncidentPermanence( incIndex )

#                indexIfIn = IndexIfItsIn( synapseToAdd, incCell )
#                if indexIfIn != None:
#                    del synapseToAdd[ indexIfIn ]

#            # 5.)...
#            if len( synapseToAdd ) > 0:
#
#                # Check to make sure this segment doesn't already have a synapse to this column.
#                realSynapsesToAdd = []
#                for synAdd in synapseToAdd:
#                    if not self.segments[ activeSeg ].AlreadySynapseToColumn( synAdd ):
#                        realSynapsesToAdd.append( synAdd )
#
#                reallyRealSynapsesToAdd = sample( realSynapsesToAdd, min( len( realSynapsesToAdd ), self.vectorMemoryDict[ "maxSynapsesToAddPer" ] ) )
#                for toAdd in reallyRealSynapsesToAdd:
# TOADD IS A CELL, IT NEEDS TO BE A COLUMN!
#                    self.AddSynapse( toAdd, activeSeg )
#
#                # If the number of synapses is above maxSynapsesPerSegment then delete random synapses.
#                while self.vectorMemoryDict[ "maxSynapsesPerSegment" ] - len( self.segments[ activeSeg ].incidentSynapses ) < 0:
#                    toDel = choice( self.segments[ activeSeg ].incidentSynapses )
#                    self.ChangePermanence( toDel, activeSeg, -1.0 )

    def CheckRelativeStimulation( self, vector ):
    # If segment is stimulated then we add the vector to relativeVector and check if it is at zero, if it is then return True.

        if self.stimulated >= 0.0:
            self.UpdateRelativeVector( vector )

            if self.Inside( self.relativeVector, [ 0 ] * len( self.relativeVector ) ):
                return True

        return False

    def CheckActivation( self, vector ):
    # Checks the incidentActivation against activationThreshold to see if segment becomes active and stimulated.

        if len( self.incidentActivation ) >= self.activationThreshold:
            self.activeAboveThresh = True

            inside = self.Inside( vector, self.vectorCenter )

            if inside:
                self.active            = True
                self.stimulated        = 1.0
#                self.segmentConfidence = len( self.incidentActivation ) * vectorScore
                self.timeSinceActive   = 0
                self.relativeVector    = [ 0 ] * len( vector )
#                self.UpdateStimulation( len( self.incidentActivation ) * vectorScore )

                return True

        self.active = False
        return False

    def GetTerminalSynapses( self ):
    # Return the terminal synapses and their permanence strengths.

        return self.terminalSynapses.copy(), self.terminalPermanences.copy()

    def RefreshSegment( self ):
    # Updates or refreshes the state of the segment.

        self.active             = False
        self.incidentActivation = []
        self.activeAboveThresh  = False
        self.stimulated -= self.vectorMemoryDict[ "segStimulatedDecay" ]

        if self.stimulated > 0.0:
            stillStimulated = True
        else:
            stillStimulated = False

        self.timeSinceActive   += 1
        if self.timeSinceActive >  self.vectorMemoryDict[ "segmentDecay" ]:
            deadTooLong = True
        else:
            deadTooLong = False

        return deadTooLong, stillStimulated

    def RemoveIncidentSynapse( self, cellToDelete ):
    # Delete the incident synapse sent.

        index = IndexIfItsIn( self.incidentSynapses, cellToDelete )
        if index != None:
            del self.incidentSynapses[ index ]
            del self.incidentPermanences[ index ]
            del self.incidentColumns[ index ]
        else:
            print( "RemoveIncidentSynapse(): Attempt to remove synapse from segment, but synapse doesn't exist." )
            exit()

    def NewIncidentSynapse( self, synapseToCreate ):
    # Create a new incident synapse.

        index = bisect_left( self.incidentColumns, synapseToCreate )

        if index == len( self.incidentColumns ):
            self.incidentColumns.append( columnToCreate )
            for incCell in range( synapseToCreate * self.vectorMemoryDict[ "cellsPerColumn" ], ( synapseToCreate * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
                self.incidentSynapses.append( synapseToCreate )
                self.incidentPermanences.append( uniform( 0, 1 ) )

        elif self.incidentSynapses[ index ] != synapseToCreate:
            self.incidentColumns.insert( index, columnToCreate )
            for incCell in range( synapseToCreate * self.vectorMemoryDict[ "cellsPerColumn" ], ( synapseToCreate * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
                self.incidentSynapses.insert( ( index * self.vectorMemoryDict[ "cellsPerColumn" ] ) + incCell, synapseToCreate )
                self.incidentPermanences.insert( index * self.vectorMemoryDict[ "cellsPerColumn" ] + incCell, uniform( 0, 1 ) )

        else:
            print( "NewIncidentSynapse(): Attempt to add synapse to segment, but synapse to column already exists." )
            exit()

    def AlreadySynapseToColumn( self, checkSynapse ):
    # Checks if this segment has a synapse to the same column as checkSynapse.

        checkColumn = int( checkSynapse / self.vectorMemoryDict[ "cellsPerColumn" ] )

        if BinarySearch( self.incidentColumns, checkColumn ):
            return True
        else:
            return False

# What I want is for the system to work, first of all. Why is it not working? It works when I increase the learning speed, but doesn't collapse when the
# speed is turned down. 

    def Equality( self, other, equalityThreshold ):
    # Runs the following comparison of equality: segment1 == segment2, comparing their activation intersection, but not vector.

        if len( FastIntersect( self.terminalSynapses, other.terminalSynapses ) ) == len( self.terminalSynapses ) and len( FastIntersect( self.incidentSynapses, other.incidentSynapses ) ) > equalityThreshold:
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

        # Working Memory portion.-----------------------------------------------
        self.numPositionCells   = 100
        self.maxPositionRange   = 800

    def HowManyActiveSegs( self ):
    # Return the number of active segments.

        return len( self.activeSegments )

#    def AddSynapse( self, FCells, incCol, segIndex ):
#    # Add an incident synapse to specified segment on specified cell.
#
#        for incCell in range( incCol * self.vectorMemoryDict[ "cellsPerColumn" ], ( incCol * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
#            Fcells[ incCell ].IncidentToThisSeg( segIndex )
#
#        self.segments[ segIndex ].NewIncidentSynapse( incCol )

#    def DeleteSynapse( self, FCells, incCell, segIndex ):
#    # Remove a synapse to specified segment from incident FCell.
#
#        # Check if synapse exists and if so delete it and all references to it in the FCell.
#
#
#        delIndex = bisect_left( self.incidentSegments[ incCell ], segIndex )
#
#        if delIndex != len( self.incidentSegments[ incCell ] ) and self.incidentSegments[ incCell ][ delIndex ] == segIndex:
#            del self.incidentSegments[ incCell ][ delIndex ]
#            del self.incidentPermanences[ incCell ][ delIndex ]
#            self.segments[ segIndex ].RemoveIncidentSynapse( incCell )
#        else:
#            print( "DeleteSynapse(): Synapse to this segment doesn't exist." )
#            exit()
#
#        # Check if segment has any synapses left. If none then mark segment for deletion.
#        if len( self.segments[ segIndex ].incidentSynapses ) == 0:
#            self.segsToDelete.append( segIndex )
#
#        return delIndex

    def DeleteSegmentsAndSynapse( self, FCells ):
    # Uses list of indices of segments that need deletion. Deletes these segments from self.segments,
    # and removes all references to them in self.incidentSegments.

        if len( self.segsToDelete ) > 0:
            for segIndex in reversed( self.segsToDelete ):
                # Delete any references to segment, and lower the index of all greater segment reference indices by one.
                for fCell in FCells:
                    fCell.DeleteIncidentSegmentReference( segIndex )

                # Decrease terminal reference.
#                if self.segments[ segIndex ].terminalSynapse != None:
#                    Cells[ self.segments[ segIndex ].terminalSynapse ].isTerminalCell -= 1

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

    def CreateSegment( self, FCells, incidentColumns, terminalColumn, vector ):
    # Creates a new segment.

        newSegment = Segment( self.vectorMemoryDict, incidentColumns, terminalColumn, vector )
        self.segments.append( newSegment )

        indexOfNew = len( self.segments ) - 1
        self.activeSegments.append( indexOfNew )
        self.stimulatedSegments.append( indexOfNew )

        # Add reference to segment in incident cells.
        for incCol in incidentColumns:
            for incCell in range( incCol * self.vectorMemoryDict[ "cellsPerColumn" ], ( incCol * self.vectorMemoryDict[ "cellsPerColumn" ] ) + self.vectorMemoryDict[ "cellsPerColumn" ] ):
                FCells[ incCell ].IncidentToThisSeg( indexOfNew )

#        for incCol in incidentColumnsList:
#            self.AddSynapse( incCol, indexOfNew )

#        if terminalCell != None:
#            Cells[ terminalCell ].isTerminalCell += 1

    def UpdateSegmentActivity( self, FCells ):
    # Make every segment that was active inactive, and refreshes its synapse activation.
    # Also add a time step to each segment, and see if it dies as a result. Delete any segments that die.
    # Also refresh FCells terminal activation.

        self.activeSegments = []

        # Increase time for all segments and deactivate them and alter their state.
        for index, segment in enumerate( self.segments ):
            deadTooLong, stillStimulated = segment.RefreshSegment()

            if deadTooLong:
                NoRepeatInsort( self.segsToDelete, index )

            if not stillStimulated:
                i = IndexIfItsIn( self.stimulatedSegments, index )
                if i != None:
                    del self.stimulatedSegments[ i ]

        for fCell in FCells:
            fCell.RefreshTerminalActivation()

        self.DeleteSegmentsAndSynapse( FCells )

#    def OCellSegmentLearning( self, activeOCells, OCells ):
#    # For the active OCells enforce this on all stimulated segments, causing them to form synapses to these if they don't have them,
#    # and supporting synapses if they exist.
#
#        for actSeg in self.activeSegments:
#            self.segments[ actSeg ].OCellLearning( OCells, activeOCells, self.vectorMemoryDict[ "initialPermanence" ], self.vectorMemoryDict[ "permanenceIncrement" ],
#                self.vectorMemoryDict[ "permanenceDecrement" ], self.vectorMemoryDict[ "maxSynapsesToAddPer" ], self.vectorMemoryDict[ "maxSynapsesPerSegment" ] )

#    def AdjustThresholds( self, lastVector, activeOCells ):
#    # Adjust the SDR threshold and vector score properties for active segments.
#
#        for activeSeg in self.activeSegments:
#            # Adjust the segments vector properties. If the segment agrees with the active OCells then support the vector position, if not then decrease.
#            self.segments[ activeSeg ].AdjustVectorProperties()
#
#            self.segments[ activeSeg ].ChangeVectorStrength( lastVector, activeOCells )
#
#        # self.segments[ activeSeg ].AdjustThreshold()

    def SegmentLearning( self, FCells, lastWinnerCells, lastActiveCells, lastVector ):
    # Perform learning on all active and inactive segments.
    # Refresh all segments then perform learning on them.
    # Delete segments that need deleting.

        # For all active segments, use the active incident cells and winner cells to modify synapses.
        print("BEGINO-------------------------------------------------------------------------------")
        for activeSeg in self.activeSegments:
            print( str(activeSeg))
            print( "Terminal Synapses: " + str(self.segments[activeSeg].terminalSynapses))
            print( "Permanence Before: " + str(self.segments[activeSeg].terminalPermanences))
            self.segments[ activeSeg ].ModifyAllSynapses( FCells )
            print( "Permanence After: " + str(self.segments[activeSeg].terminalPermanences))
        print("ENDO-------------------------------------------------------------------------------")

#        self.AdjustThresholds( lastVector, activeOCells )

#        self.OCellSegmentLearning( activeOCells, OCells )

#        self.CheckIfSegsIdentical( FCells )

#    def GetStimulatedOCells( self, numberOCells ):
#    # Return the OCells count from all stimulated segments by summing their permances.
#
#        OCellCounts = [ 0.0 ] * numberOCells
#
#        for stimSeg in self.stimulatedSegments:
#            for oCell in self.segments[ stimSeg ].ReturnOCellSynapses():
#                OCellCounts[ oCell[ 0 ] ] += oCell[ 1 ]
#
#        return OCellCounts

#    def ResetStimulatedSegments( self ):
#    # Turn off all stimulated segments.
#
#        for stimSeg in self.stimulatedSegments:
#            self.segments[ stimSeg ].RefreshSegment( self.vectorMemoryDict[ "segmentDecay" ], True )
#
#        self.stimulatedSegments = []

    def StimulateSegments( self, FCells, activeCells, vector ):
    # Using the activeCells and next vector find all segments that activate from this.
    # Use these active segments to get the predicted cells for given column.

        predictedCells = []
        potentialSegs  = []             # The segments that are stimulated and at zero vector.

        # First stimulate and activate any segments from the incident cells.
        # Activate the synapses in segments using activeCells.
        for incCell in activeCells:
            incCol, incSegments = FCells[ incCell ].ReturnIncidentOn()
            for entry in incSegments:
                self.segments[ entry ].IncidentCellActive( incCell, incCol )
        # Check the overlap of all segments and see which ones are active, and add the terminalCell to stimulatedCells.
        for segIndex, segment in enumerate( self.segments ):
            adEm = False

            if segment.CheckActivation( vector ):
                NoRepeatInsort( self.activeSegments, segIndex )
                NoRepeatInsort( self.stimulatedSegments, segIndex )
                adEm = True
            elif segment.CheckRelativeStimulation( vector ):
                adEm = True

            if adEm:
                terminalCells, terminalPermanences = segment.GetTerminalSynapses()
                for index in range( len( terminalCells ) ):
                    # Terminal cell becomes predictive.
                    NoRepeatInsort( predictedCells, terminalCells[ index ] )
                    # Add strimulation to the cell.
                    FCells[ terminalCells[ index ] ].AddTerminalStimulation( terminalPermanences[ index ] )

        return predictedCells

#    def UpdateStimSegmentsVector(self, vector ):
#    # Uses vector change to update all stimulated segments relativeVector.
#
#        for stimSeg in self.stimulatedSegments:
#            self.segments[ stimSeg ].UpdateRelativeVector( vector )

#    def GetSegmentAverages( self, FCells ):
#    # Gets the active segments which terminate on a winner cell and gathers their vector data for working memory.
#
#        averages = [ 0, [ 0.0 ] * self.vectorMemoryDict[ "vectorDimensions" ], [ 0.0 ] * self.vectorMemoryDict[ "vectorDimensions" ], 0 ]          # [ Segment count, Sum of standardDeviation, Sum of vectorConfidence, Sum of activation thresholds ]
#
#        for activeSeg in self.activeSegments:
#            if FCells[ self.segments[ activeSeg ].terminalSynapse ].winner:
#                averages[ 0 ] += 1
#                for d in range( self.vectorMemoryDict[ "vectorDimensions" ] ):
#                    averages[ 1 ][ d ] += self.segments[ activeSeg ].standardDeviation[ d ]
#                    averages[ 2 ][ d ] += self.segments[ activeSeg ].vectorConfidence[ d ]
#                averages[ 3 ] += self.segments[ activeSeg ].activationThreshold
#
#        return averages

    def ChangePermanence( self, FCells, incCell, segIndex, permanenceChange ):
    # Change the permanence of synapse incident on incCell, part of segIndex, by permanenceChange.

        if FCells[ incCell ].ConnectionToSegment( segIndex ):
            self.segments.ModifyIncidentSynapse( incCell, permanenceChange )
        else:
            print("ChangePermanence(): Referenced cell does not have incident connection to referenced segment.")
            exit()

    def ThereCanBeOnlyOne( self, FCells, activeCellsList ):
    # Choose the winner cell for column by choosing active one with highest terminal activation.

        # Find the predicted cell
        greatestActivation = 0
        greatestCell       = activeCellsList[ 0 ]

        if len( activeCellsList ) > 1:
            for cell in activeCellsList:
                if FCells[ cell ].GetTerminalActivation() > greatestActivation:
                    greatestActivation = FCells[ cell ].GetTerminalActivation()
                    greatestCell       = cell

        return greatestCell

#    def ReturnActivation( self, seg ):
#    # Returns the sum of permanences of synapses incident on incCellList and part of seg.
#
#        activation = 0.0
#
#        for incCol in self.segments[ seg ].incidentColumns:
#            entryIndex = IndexIfItsIn( self.incidentSegments[ incCol ], seg )
#            if entryIndex != None:
#                activation += self.incidentPermanences[ incCol ][ entryIndex ]
#
#        return activation

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
                    NoRepeatInsort( self.segsToDelete, group[ segIndex ] )

# ------------------------------------------------------------------------------

class FCell:

    def __init__( self, colID ):
    # Create a new feature level cell with synapses to OCells.

        # FCell state variables.
        self.active     = False
        self.lastActive = False
        self.predicted  = False
        self.winner     = False
        self.lastWinner = False

        self.column     = colID

        self.asIncident = []            # Keeps track of what segments this cell is incident on.

        self.terminalActivation = 0.0

#        self.isTerminalCell = 0           # Number of segments this cell is terminal on.

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

    def IncidentToThisSeg( self, segIndex ):
    # Add reference that this cell is on this segment.

        if BinarySearch( self.asIncident, segIndex ):
            print( "IncidentToThisSeg(): Tried to add reference to segment, but already exists." )
            exit()
        else:
            NoRepeatInsort( self.asIncident, segIndex )

    def DeleteIncidentSegmentReference( self, segIndex ):
    # Checks if this cell is incident to this segment, if so then delete it. Also lower all segment references by one.

        if len( self.asIncident ) > 0:
            indexAt = bisect_left( self.asIncident, segIndex )

            if indexAt < len( self.asIncident ) and self.asIncident[ indexAt ] == segIndex:
                del self.asIncident[ indexAt ]

            while indexAt < len( self.asIncident ):
                self.asIncident[ indexAt ] -= 1
                indexAt += 1

    def ReturnIncidentOn( self ):
    # Returns this cells column, and a list of segments this cell is incident on.

        return self.column, self.asIncident.copy()

    def ConnectionToSegment( segIndex ):
    # Returns true if this cell is incident on segment segIndex, otherwise returns False.

        return BinarySearch( self.asIncident, segIndex )

    def RefreshTerminalActivation( self ):
    # Refreshes terminalActivation back to 0.0

        self.terminalActivation = 0.0

    def AddTerminalStimulation( self, stimAdd ):
    # Adds permanence value to terminalActivation.

        self.terminalActivation += stimAdd

    def GetTerminalActivation( self ):
    # Returns terminalActivation.

        return self.terminalActivation

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

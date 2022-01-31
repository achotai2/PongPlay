from bisect import bisect_left
from random import uniform, choice, sample
import numpy as np

def BinarySearch( list, val ):
# Search a sorted list: return False if val not in list, and True if it is.

    i = bisect_left( list, val )

    if i != len( list ) and list[ i ] == val:
        return True

    else:
        return False

def IndexIfItsIn( list, val ):
# Returns the left-most index of val in list, if it's there. If val isn't in list then return None.

    i = bisect_left( list, val )

    if i != len( list ) and list[ i ] == val:
        return i

    else:
        return None

def NoRepeatInsort( list, val ):
# Inserts item into list in a sorted way, if it does not already exist (no repeats).
# Returns the index where it inserted val into list.

    idx = bisect_left( list, val )

    if idx == len( list ):
        list.append( val )
        return ( len( list ) - 1 )

    elif list[ idx ] != val:
        list.insert( idx, val )
        return idx

def RepeatInsort( list, val ):
# Inserts item into list in a sorted way (allow repeats).
# Returns the index where it inserted val into list.

    idx = bisect_left( list, val )

    if idx == len( list ):
        list.append( val )
        return ( len( list ) - 1 )

    else:
        list.insert( idx, val )
        return idx

def CheckInside( vector1, vector2, checkRange ):
# Checks if given position is inside range.

    if len( vector1 ) != len( vector2 ):
        print( "Error: Vectors not of same length." )
        exit()

    for i in range( len( vector1 ) ):
        if vector1[ i ] > vector2[ i ] + checkRange or vector1[ i ] < vector2[ i ] - checkRange:
            return False

    return True

#-------------------------------------------------------------------------------

class Segment:

    def __init__( self, incidentCellsList, initialPermanence, vector, posRange, minActivation ):
    # Create a new OSegment, which connects multiple OCells to multiple FCells in same column. One will be chosen
    # as winner eventually.
    # Within this create an FSegment, with multiple synapses to FCells from these FCells.
    # Also a vector which will be used to activate segment.

        self.active              = True      # True means it's predictive, and above threshold terminal cells fired.
        self.lastActive          = False     # True means it was active last time step.
        self.lastOverlapScore    = 0         # The overlap score that this segment last attained.
        self.timeSinceActive     = 0         # Time steps since this segment was active last.
        self.activationThreshold = minActivation

        # FSegment portion--------
        self.incidentSynapses    = incidentCellsList.copy()
        self.incidentPermanences = [ initialPermanence ] * len( incidentCellsList )

        # Vector portion---------
        self.dimensions = len( vector )      # Number of dimensions vector positions will be in.
        self.vector     = []                 # A list of tuples, one for each dimension, with a range for location.
        for i in range( self.dimensions ):
            self.vector.append( ( vector[ i ] - posRange / 2, vector[ i ] + posRange / 2 ) )

#    def AddStateToRecord( self ):
#    # Adds details of current state, as a string, to record to be printed upon program completion.
#
#        primedSegString = []
#        for index, pSeg in enumerate( self.toPrime ):
#            primedSegString.append( ( pSeg, self.toPrimePermanences[ index ] ) )
#
#        return ( "< Vector: %s, \n Active: %s, Last Active: %s, Time Since Active: %s, Activation Threshold: %s \n Active Incident Synapses: %s, %s, \n Active Terminal Synapses: %s, %s \n Primed Connections: %s > \n"
#            % (self.vector, self.active, self.lastActive, self.timeSinceActive, self.activationThreshold, self.incidentSynapses, self.incidentPermanences, self.terminalSynapses, self.terminalPermanences, primedSegString ) )

    def Equality( self, other, equalityThreshold ):
    # Runs the following comparison of equality: segment1 == segment2.

        incidentEquality = 0

        otherIndex       = 0
        selfIndex        = 0

        otherCenterVector = []
        for dim1 in other.vector:
            otherCenterVector.append( ( dim1[ 0 ] + dim1[ 1 ] / 2 ) )
        selfCenterVector = []
        for dim2 in self.vector:
            selfCenterVector.append( ( dim2[ 0 ] + dim2[ 1 ] / 2 ) )

        if self.dimensions == other.dimensions and self.Inside( otherCenterVector ) and other.Inside( selfCenterVector ):
            while otherIndex < len( other.incidentSynapses ) and selfIndex < len( self.incidentSynapses ):
                if other.incidentSynapses[ otherIndex ] == self.incidentSynapses[ selfIndex ]:
                    incidentEquality += 1
                    otherIndex += 1
                    selfIndex += 1

                elif other.incidentSynapses[ otherIndex ] < self.incidentSynapses[ selfIndex ]:
                    otherIndex += 1

                elif other.incidentSynapses[ otherIndex ] > self.incidentSynapses[ selfIndex ]:
                    selfIndex += 1

        if incidentEquality > equalityThreshold:
            return True
        else:
            return False

    def Inside( self, vector ):
    # Checks if given position is inside range.

        for i in range( self.dimensions ):
            if vector[ i ] < self.vector[ i ][ 0 ] or vector[ i ] > self.vector[ i ][ 1 ]:
                return False

        return True

    def CheckOverlap( self, FCellListBool, cellsPerColumn, lowerThreshold ):
    # Check FColumnList for overlap with incidentSynapses. If overlap is above threshold return True, otherwise False.

        self.lastOverlapScore = 0

        for synapse in self.incidentSynapses:
            if FCellListBool[ synapse ]:
                self.lastOverlapScore += 1

        if self.lastOverlapScore >= self.activationThreshold:
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

    def DecayAndCreate( self, lastActiveCellsBool, lastWinnerCells, permanenceIncrement, permanenceDecrement, initialPermanence, maxSynapsesToAddPer, maxSynapsesPerSegment, cellsPerColumn ):
    # Increase synapse strength to lastActive cells that already have synapses.
    # Decrease synapse strength to not lastActive cells that have synapses.
    # Build new synapses to lastActive cells that don't have synapses, but only one cell per column can have synapses.
    # Delete synapses whose permanences have decayed to zero.

# DOES THIS REALLY NEED LASTACTIVECELL BOOL AND LASTWINNERCELLS? UNLESS THE COLUMN IS BURSTING THEN THESE TWO SHOULD BE EQUAL
# NOW THAT WE'VE MADE ONLY ONE ACTIVE CELL PER COLUMN.

        synapseToDelete = []
        synapseToAdd    = []

        synIndex = 0
        for cellIndex, lCellActive in enumerate( lastActiveCellsBool ):

            # Increase synapse strength to lastActive cells that already have synapses.
            if lCellActive and synIndex < len( self.incidentSynapses ) and self.incidentSynapses[ synIndex ] == cellIndex:
                self.incidentPermanences[ synIndex ] += permanenceIncrement
                if self.incidentPermanences[ synIndex ] > 1.0:
                    self.incidentPermanences[ synIndex ] = 1.0
                synIndex += 1

            # Decrease synapse strength to not lastActive cells that have synapses.
            elif not lCellActive and synIndex < len( self.incidentSynapses ) and self.incidentSynapses[ synIndex ] == cellIndex:
                self.incidentPermanences[ synIndex ] -= permanenceDecrement
                if self.incidentPermanences[ synIndex ] < 0.0:
                    synapseToDelete.insert( 0, synIndex )
                synIndex += 1

            # Build new synapses to lastActive cells that don't have synapses.
            # Only one cell per column can have synapses, so check for this first.
            elif lCellActive:
                addNewSynapse = False

                if BinarySearch( lastWinnerCells, cellIndex ):
                    thisColumn = int( cellIndex / cellsPerColumn )
                    if synIndex == 0:
                        if int( self.incidentSynapses[ synIndex ] / cellsPerColumn ) != thisColumn:
                            addNewSynapse = True
                    elif synIndex < len( self.incidentSynapses ):
                        if int( self.incidentSynapses[ synIndex ] / cellsPerColumn ) != thisColumn and int( self.incidentSynapses[ synIndex - 1 ] / cellsPerColumn ) != thisColumn:
                            addNewSynapse = True
                    elif synIndex == len( self.incidentSynapses ):
                        if int( self.incidentSynapses[ synIndex - 1 ] / cellsPerColumn ) != thisColumn:
                            addNewSynapse = True

                if addNewSynapse:
                    synapseToAdd.append( cellIndex )
                    synIndex += 1

        # Delete synapses marked for deletion.
        for toDel in synapseToDelete:
            del self.incidentSynapses[ toDel ]
            del self.incidentPermanences[ toDel ]

        realSynapsesToAdd = sample( synapseToAdd, min( len( synapseToAdd ), maxSynapsesToAddPer ) )
        for toAdd in realSynapsesToAdd:
            index = bisect_left( self.incidentSynapses, toAdd )
            self.incidentSynapses.insert( index, toAdd )
            self.incidentPermanences.insert( index, toAdd )

        # If the number of synapses is above maxSynapsesPerSegment then delete the ones with lowest synapses.
        while maxSynapsesPerSegment - len( self.incidentSynapses ) < 0:
            minValue = min( self.incidentPermanences )
            minIndex = self.incidentPermanences.index( minValue )
            del self.incidentSynapses[ minIndex ]
            del self.incidentPermanences[ minIndex ]

# ------------------------------------------------------------------------------

class FCell:

    def __init__( self, initialPermanence, numOCells, pctAllowedOCellConns, segmentDecay ):
    # Create a new feature level cell with synapses to OCells.

        self.active     = False
        self.lastActive = False
        self.predicted  = False
        self.winner     = False
        self.lastWinner = False
        self.primed     = False

        self.activeCount = 0
        self.states      = []
        self.statesCount = []

        self.segments     = []
        self.segmentDecay = segmentDecay

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

    def OCellConnect( self, totalActiveOCells, permanenceIncrement, permanenceDecrement ):
    # Look through the ordered indices of totalOCellConnections. Choose the lowest one in our OCellConnections
    # to remove, and the highest one not in our OCellConnections to add.

        for index, oCell in enumerate( self.OCellConnections ):
            if BinarySearch( totalActiveOCells, oCell ):
                self.OCellPermanences[ index ] += permanenceIncrement

                if self.OCellPermanences[ index ] > 1.0:
                    self.OCellPermanences[ index ] = 1.0
            else:
                self.OCellPermanences[ index ] -= permanenceDecrement

                if self.OCellPermanences[ index ] < 0.0:
                    self.OCellPermanences[ index ] = 0.0

    def NumSegments( self ):
    # Return the number of segments terminating on this cell.

        return len( self.segments )

    def IncrementSegTime( self, segsToDeleteList ):
    # Increment every OSegments timeSinceActive.

        for segIndx in range( len( self.segments ) ):
            self.segments[ segIndx ].timeSinceActive += 1

            if self.segments[ segIndx ].timeSinceActive > self.segmentDecay:
                NoRepeatInsort( segsToDeleteList, segIndx )

    def DeleteSegments( self, segsToDeleteList ):
    # Deletes all segments in segsToDeleteList.

        for index in reversed( segsToDeleteList ):
            del self.segments[ index ]

    def UpdateSegmentActivity( self ):
    # Make every segment that was active into lastActive, and every lastActive into not active.

        for segment in self.segments:
            if segment.lastActive:
                segment.lastActive = False

            if segment.active:
                segment.active     = False
                segment.lastActive = True

    def HighestOverlapForActiveSegment( self ):
    # For all active segments on this cell, returns the the one with the highest lastOverlapScore.

        highestOverlap = 0

        for seg in self.segments:
            if seg.active and seg.lastOverlapScore > highestOverlap:
                highestOverlap = seg.lastOverlapScore

        return highestOverlap

    def CheckIfPredicted( self, activeFCellsBool, cellsPerColumn, vector, lowerThreshold ):
    # Checks if this cell has any segments predicting by checking incident overlap and vector.
    # If so then activate that segment and make cell predicted, otherwise not predicted.

# SHOULD IMPROVE THIS BY CHECKING IF MULTIPLE SEGMENTS ARE ACTIVE WE ONLY WANT THE ONE WITH THE MOST OVERLAP MAYBE.
# MAYBE COMBINE THE TWO INTO ONE, OR DELETE ONE, IF ACTIVE TOGETHER OFTEN.

# CAN ALSO IMPROVE BY MODIFYING SEGMENTS SO WE DONT HAVE TO CHECK EACH SEGMENT SEPARATELY. CAN INSTEAD GO THROUGH ALL THE
# FCELLS ONCE AND CHECK ALL SEGMENTS AT ONCE. IT SHOULD BE POSSIBLE TO DO THIS.

        anySegments = False

        for segment in self.segments:
            if segment.Inside( vector ) and segment.CheckOverlap( activeFCellsBool, cellsPerColumn, lowerThreshold ):
                segment.active = True
                anySegments    = True

        if anySegments:
            self.predicted = True
        else:
            self.predicted = False

    def CheckIfSegsIdentical( self, segsToDelete, activeSegs, equalityThreshold ):
    # Compares all active segments to see if they have identical vectors and active synapse bundles.
    # If any do then delete one of them.

        if len( activeSegs ) > 1:
            for index1 in range( len( activeSegs ) - 1 ):
                for index2 in range( index1 + 1, len( activeSegs ) ):
                    actSeg1 = activeSegs[ index1 ]
                    actSeg2 = activeSegs[ index2 ]

                    if actSeg1 != actSeg2:
                        if self.segments[ actSeg1 ].Equality( self.segments[ actSeg2 ], equalityThreshold ):
                            NoRepeatInsort( segsToDelete, actSeg2 )
                    else:
                        print( "Error in CheckIfSegsIdentical()")
                        exit()

    def CreateSegment( self, vector, incidentCellWinners, initialPermanence, initialPosVariance, FActivationThresholdMin, cellsPerColumn  ):
    # Create a new Segment, terminal on these active columns, incident on last active columns.

        newSegment = Segment( incidentCellWinners, initialPermanence, vector, initialPosVariance, FActivationThresholdMin )
        self.segments.append( newSegment )

    def SegmentLearning( self, lastVector, lastActiveFCellsBool, incidentCellWinners, initialPermanence, initialPosVariance,
        FActivationThresholdMin, FActivationThresholdMax, permanenceIncrement, permanenceDecrement,
        maxNewFToFSynapses, maxSynapsesPerSegment, cellsPerColumn, equalityThreshold ):
    # Perform learning on segments, and create new ones if neccessary.

        segsToDelete = []

        # Add time to all segments, and delete segments that haven't been active in a while.
        self.IncrementSegTime( segsToDelete )

        # If this cell is winner but the cell is not predicted then create a new segment to the lastActive FCells.
        if self.winner and not self.predicted:
            self.CreateSegment( lastVector, incidentCellWinners, initialPermanence, initialPosVariance, FActivationThresholdMin, cellsPerColumn )

        # For every active and lastActive segment...
        activeSegs = []
        for index, segment in enumerate( self.segments ):
            if segment.active:
                activeSegs.append( index )
                # If active segments have positive synapses to non-active columns then decay them.
                # If active segments do not have terminal or incident synapses to active columns create them.
                segment.DecayAndCreate( lastActiveFCellsBool, incidentCellWinners, permanenceIncrement, permanenceDecrement,
                    initialPermanence, maxNewFToFSynapses, maxSynapsesPerSegment, cellsPerColumn )

                # Adjust the segments activation threshold depending on number of winners selected.
                segment.AdjustThreshold( FActivationThresholdMin, FActivationThresholdMax )

        # If there is more than one segment active check if they are idential, if so delete one.
        self.CheckIfSegsIdentical( segsToDelete, activeSegs, equalityThreshold )

        # Delete segments that had all their incident or terminal synapses removed.
        self.DeleteSegments( segsToDelete )

# ------------------------------------------------------------------------------

class OCell:

    def __init__( self ):
    # Create a new cell in object layer.

        self.active = False

        self.segments = []

    def Deactivate( self ):
    # Deactivate this cell, and all segments.

        self.active = False
        for seg in self.segments:
            seg.active = False

    def CheckOverlapAndActivate( self, predictedActiveCellsBool, cellsPerColumn, lowerThreshold ):
    # Check this inactive OCells's segments for overlap with predictedActiveCells and see if it become active.

        toReturn = False

        for seg in self.segments:
            if seg.CheckOverlap( predictedActiveCellsBool, cellsPerColumn, lowerThreshold ):
                seg.active  = True
                self.active = True
                toReturn = True

        return toReturn

    def OCellLearning( self, predictedActiveCells, predictedActiveCellsBool, cellsPerColumn, lowerThreshold, permanenceIncrement, permanenceDecrement, initialPermanence, maxSynapsesToAddPer, maxSynapsesPerSegment, minActivation ):
    # Check this active OCell if it has segments which activate with given predictedActiveCells.
    # If they do then activate and strengthen segment, perform learning on it, DecayAndCreate.
    # If they don't then create a new segment.

        activeSeg = False
        for seg in self.segments:
            if seg.CheckOverlap( predictedActiveCellsBool, cellsPerColumn, lowerThreshold ):
                seg.active = True
                activeSeg  = True
                seg.DecayAndCreate( predictedActiveCellsBool, predictedActiveCells, permanenceIncrement, permanenceDecrement, initialPermanence, maxSynapsesToAddPer, maxSynapsesPerSegment, cellsPerColumn )

        if not activeSeg:
            self.segments.append( Segment( predictedActiveCells, initialPermanence, [], 0, minActivation ) )

    def ReturnGreatestOverlapScore( self ):
    # Returns the greatest last overlap score of all active segments.

        greatestOverlapScore = 0

        for seg in self.segments:
            if seg.active:
                if seg.lastOverlapScore > greatestOverlapScore:
                    greatestOverlapScore = seg.lastOverlapScore

        return greatestOverlapScore

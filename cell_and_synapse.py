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

def FastOverlap( list1, list2, threshold, stopIfThreshold ):
# Checks if overlap between two sorted lists is above or equal to threshold or not. Returns overlap as integer.
# If stopIfThreshold = True then function will stop if threshold overlap is reached, or if not possible to be reached.

    i       = 0
    j       = 0
    overlap = 0

    while i < len( list1 ) and j < len( list2 ):
        if list1[ i ] < list2[ j ]:
            i += 1
        elif list1[ i ] > list2[ j ]:
            j += 1
        else:
            overlap += 1
            i += 1
            j += 1

        if stopIfThreshold and ( overlap >= threshold or overlap + min( len( list1 ) - i, len( list2 ) - j ) < threshold ):
            break

    return overlap

#-------------------------------------------------------------------------------

class Segment:

    def __init__( self, incidentCellsList, initialPermanence, vector, posRange, minActivation ):
    # Create a new OSegment, which connects multiple OCells to multiple FCells in same column. One will be chosen
    # as winner eventually.
    # Within this create an FSegment, with multiple synapses to FCells from these FCells.
    # Also a vector which will be used to activate segment.

        self.active              = True             # True means it's predictive, and above threshold terminal cells fired.
        self.lastActive          = False            # True means it was active last time step.
        self.excited             = False            # True means it was active some time recently.
        self.lastOverlapScore    = 0                # The overlap score that this segment last attained.
        self.timeSinceActive     = 0                # Time steps since this segment was active last.
        self.activationThreshold = minActivation    # The minimum overlap for segment to become active.

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

    def CheckOverlap( self, cellList ):
    # Check FColumnList for overlap with incidentSynapses. If overlap is above threshold return True, otherwise False.

        self.lastOverlapScore = FastOverlap( self.incidentSynapses, cellList, self.activationThreshold, True )

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

    def DecayAndCreate( self, activeCells, winnerCells, permanenceIncrement, permanenceDecrement, initialPermanence, maxSynapsesToAddPer, maxSynapsesPerSegment, cellsPerColumn, careAboutColumns ):
    # Increase synapse strength to cells that already have synapses.
    # Decrease synapse strength to not cells that have synapses.
    # Build new synapses to cells that don't have synapses, but only one cell per column can have synapses.
    # Delete synapses whose permanences have decayed to zero.

        synapseToDelete = []
        synapseToAdd    = []

        synIndex = 0
        actIndex = 0
        while synIndex < len( self.incidentSynapses ) or actIndex < len( activeCells ):

            # Increase synapse strength to active cells that already have synapses.
            if synIndex < len( self.incidentSynapses ) and actIndex < len( activeCells ) and self.incidentSynapses[ synIndex ] == activeCells[ actIndex ]:
                self.incidentPermanences[ synIndex ] += permanenceIncrement
                if self.incidentPermanences[ synIndex ] > 1.0:
                    self.incidentPermanences[ synIndex ] = 1.0

                if synIndex < len( self.incidentSynapses ):
                    synIndex += 1
                if actIndex < len( activeCells ):
                    actIndex += 1

            # Decrease synapse strength to not active cells that have synapses.
            elif synIndex < len( self.incidentSynapses ) and ( actIndex == len( activeCells ) or self.incidentSynapses[ synIndex ] < activeCells[ actIndex ] ):
                self.incidentPermanences[ synIndex ] -= permanenceDecrement
                if self.incidentPermanences[ synIndex ] < 0.0:
                    synapseToDelete.insert( 0, synIndex )

                if synIndex < len( self.incidentSynapses ):
                    synIndex += 1

            # Build new synapses to cells that don't have synapses.
            # If careAboutColumns is True then only one cell per column can have synapses, so check for this first.
            else:
                # Check if this is a winnerCell, since if the col is bursting all the cells will be active on it.
                if BinarySearch( winnerCells, activeCells[ actIndex ] ):
                    thisColumn = int( activeCells[ actIndex ] / cellsPerColumn )
                    # Check if there is already a synapse to this column.
                    alreadyExists = False
                    if careAboutColumns:
                        for ins in self.incidentSynapses:
                            if int( ins / cellsPerColumn ) == thisColumn:
                                alreadyExists = True
                    if not alreadyExists:
                        synapseToAdd.append( activeCells[ actIndex ] )

                if actIndex < len( activeCells ):
                    actIndex += 1

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

    def ReturnIncidentCells( self ):
    # Return as a list the incident cells for this segment.

        return self.incidentSynapses.copy()

    def IncrementTime( self, maxTimeSinceActive ):
    # Up my seg time.

        self.timeSinceActive += 1
        if self.timeSinceActive >= maxTimeSinceActive:
            return True
        else:
            return False

# ------------------------------------------------------------------------------

class FCell:

    def __init__( self, initialPermanence, numOCells ):
    # Create a new feature level cell with synapses to OCells.

        self.active     = False
        self.predicted  = False
        self.winner     = False
        self.primed     = False

        self.activeCount = 0
        self.states      = []
        self.statesCount = []

        self.segments     = []

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

    def NumSegments( self ):
    # Return the number of segments terminating on this cell.

        return len( self.segments )

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

    def CheckIfPredicted( self, myIndex, synapseList, predictionThreshold ):
    # Checks if this cell has any segments predicting by checking incident overlap and vector.
    # If so then activate that segment and make cell predicted, otherwise not predicted.

        self.predicted = False

        for seg in self.segments:
            cellsOnSynapse = seg.ReturnIncidentCells()
            overlap = 0
            for cell in cellsOnSynapse:
                if synapseList[ cell ][ myIndex ]:
                    overlap += 1

            if overlap >= predictionThreshold:
                self.predicted = True
                return True

        return False

    def UpdateCellState( self ):
    # Make inactive and not winner.

        self.active = False
        self.winner = False

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

    def SegmentLearning( self, activeOCells, initialPermanence, FActivationThresholdMin, FActivationThresholdMax, maxTimeSinceActive,
        permanenceIncrement, permanenceDecrement, maxSynapsesToAddPer, maxSynapsesPerSegment, cellsPerColumn, equalityThreshold ):
    # Perform learning on segments, and create new ones if neccessary.

        segsToDelete = []

        # Add time to all segments, and delete segments that haven't been active in a while.
        for segIndex, seg in enumerate( self.segments ):
            if seg.IncrementTime( maxTimeSinceActive ):
                segsToDelete[ segIndex ]

        # If this cell is active then check it against activeOCells to see if it has any matching segments.
        if self.active:
            activeSegs = []
            for index, seg in enumerate( self.segments ):
                if seg.CheckOverlap( activeOCells ):
                    activeSegs.append( index )

                    # If so then perform learning on it.
                    seg.DecayAndCreate( activeOCells, activeOCells, permanenceIncrement, permanenceDecrement, initialPermanence, maxSynapsesToAddPer, maxSynapsesPerSegment, cellsPerColumn, False )

                    # Adjust the segments activation threshold depending on number of winners selected.
                    seg.AdjustThreshold( FActivationThresholdMin, FActivationThresholdMax )

            # If no segments align with active OCells then create a new segment.
            if len( activeSegs ) == 0:
                self.segments.append( Segment( activeOCells, initialPermanence, [], 0, FActivationThresholdMin ) )

            # If there is more than one segment active check if they are idential, if so delete one.
            self.CheckIfSegsIdentical( segsToDelete, activeSegs, equalityThreshold )

        # Delete segments that had all their incident or terminal synapses removed.
        self.DeleteSegments( segsToDelete )

# ------------------------------------------------------------------------------

class OCell:

    def __init__( self ):
    # Create a new cell in object layer.

        self.active = False

        self.locations = [ [ 0, 0 ] ]

        self.segments = []

    def Deactivate( self ):
    # Deactivate this cell, and all segments.

        self.active = False
        for seg in self.segments:
            seg.active = False

        self.locations = [ [ 0, 0 ] ]

    def UpdateSegmentActivity( self ):
    # Turns all active segments to lastActive, and lastActive to inactive.

        for seg in self.segments:
            if seg.active:
                seg.active     = False
                seg.lastActive = True
            else:
                seg.lastActive = False

    def UpdateVector( self, vector ):
    # Update the local hypothesis of what location we are looking at on our object.

        if self.active:
            for loc in self.locations:
                if len( vector ) != len( loc ):
                    print( "Vector sent to OCell not of right length." )
                    exit()

                for index in range( len( loc ) ):
                    loc[ index ] += vector[ index ]

    def CheckSegmentActivation( self, activeFCells ):
    # Check this active OCells's segments for overlap with activeFCells and see if they become active.
    # First though check if the vector fits that segment.

        if self.active:
            for seg in self.segments:
                for loc in self.locations:
                    if seg.Inside( loc ) and seg.CheckOverlap( activeFCells ):
                        seg.active  = True
                        seg.excited = True

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

    def DeleteSegments( self, segsToDeleteList ):
    # Deletes all segments in segsToDeleteList.

        for index in reversed( segsToDeleteList ):
            del self.segments[ index ]

    def SegmentLearning( self, activeFCells, winnerFCells, permanenceIncrement, permanenceDecrement, initialPermanence,
        maxSynapsesToAddPer, maxSynapsesPerSegment, cellsPerColumn, FActivationThresholdMin, FActivationThresholdMax,
        initialPosVariance, maxTimeSinceActive, equalityThreshold ):
    # Check this active OCell if it has segments which have been activated.
    # If they do then activate and strengthen segment, perform learning on it (DecayAndCreate).
    # If they don't then create a new segment.

        segsToDelete = []

        # Add time to all segments, and delete segments that haven't been active in a while.
        for segIndex, seg in enumerate( self.segments ):
            if seg.IncrementTime( maxTimeSinceActive ):
                segsToDelete.append( segIndex )

# SHOULD BRING IN SEGS TO DELETE, TIME SINCE LAST ACTIVE, LIKE WHAT WE HAVE WITH FCELLS NOW, AND ADJUST OVERLAP THRESHOLD.
        if self.active:
            activeSegs = []
            for index, seg in enumerate( self.segments ):
                if seg.active:
                    activeSegs.append( index )
                    # Perform learning on segment.
                    seg.DecayAndCreate( activeFCells, winnerFCells, permanenceIncrement, permanenceDecrement, initialPermanence, maxSynapsesToAddPer, maxSynapsesPerSegment, cellsPerColumn, True )

                    # Adjust the segments activation threshold depending on number of winners selected.
                    seg.AdjustThreshold( FActivationThresholdMin, FActivationThresholdMax )

            if len( activeSegs ) == 0:
# WHAT VECTOR DO WE WANT TO CREATE IT WITH? DO WE CREATE A SEGMENT WITH ALL VECTORS IF MULTIPLE? OR DON"T CREATE IF MULTIPLE?
                self.segments.append( Segment( activeFCells, initialPermanence, self.locations[ 0 ], initialPosVariance, FActivationThresholdMin ) )

            # If there is more than one segment active check if they are idential, if so delete one.
            self.CheckIfSegsIdentical( segsToDelete, activeSegs, equalityThreshold )

        # Delete segments that had all their incident or terminal synapses removed.
        self.DeleteSegments( segsToDelete )

    def GetPredicted( self, vector, numberOfFCells ):
    # Using the vector get what FCells are predicted.

        FCellsBool = [ False ] * numberOfFCells

        if self.active:

            for seg in self.segments:
                for loc in self.locations:
                    if len( vector ) != len( loc ):
                        print( "Vector sent to OCell not of right length." )
                        exit()

                    if seg.Inside( [ loc[ i ] + vector[ i ] for i in range( len( loc ) ) ] ):
                        FCellsList = seg.ReturnIncidentCells()

                        for fCell in FCellsList:
                            FCellsBool[ fCell ] = True

        return FCellsBool

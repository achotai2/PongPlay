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
        self.lastActive          = False      # True means it was active last time step.
        self.timeSinceActive     = 0          # Time steps since this segment was active last.
        self.activationThreshold = minActivation

        # FSegment portion--------
        self.incidentSynapses    = incidentCellsList.copy()
        self.incidentPermanences = [ initialPermanence ] * len( incidentCellsList )

        # Vector portion---------
        self.dimensions = len( vector )      # Number of dimensions vector positions will be in.
        self.vector     = []                 # A list of tuples, one for each dimension, with a range for location.
        for i in range( self.dimensions ):
            posI = ( vector[ i ] - posRange / 2, vector[ i ] + posRange / 2 )
            self.vector.append( posI )

    def AddStateToRecord( self ):
    # Adds details of current state, as a string, to record to be printed upon program completion.

        primedSegString = []
        for index, pSeg in enumerate( self.toPrime ):
            primedSegString.append( ( pSeg, self.toPrimePermanences[ index ] ) )

        return ( "< Vector: %s, \n Active: %s, Last Active: %s, Time Since Active: %s, Activation Threshold: %s \n Active Incident Synapses: %s, %s, \n Active Terminal Synapses: %s, %s \n Primed Connections: %s > \n"
            % (self.vector, self.active, self.lastActive, self.timeSinceActive, self.activationThreshold, self.incidentSynapses, self.incidentPermanences, self.terminalSynapses, self.terminalPermanences, primedSegString ) )

    def Equality( self, other, equalityThreshold, lowerThreshold ):
    # Runs the following comparison of equality: segment1 == segment2.

        incidentEquality = 0
        terminalEquality = 0

        if self.dimensions == other.dimensions and self.vector == other.vector :

                for i in range( len( self.incidentSynapses ) ):
                    if ( self.incidentSynapses[ i ].getWinnerCell()[ 0 ] == other.incidentSynapses[ i ].getWinnerCell()[ 0 ]
                        and self.incidentSynapses[ i ].getWinnerCell()[ 1 ] > lowerThreshold
                        and other.incidentSynapses[ i ].getWinnerCell()[ 1 ] > lowerThreshold ):

                            incidentEquality += 1

                for ii in range( len( self.terminalSynapses ) ):
                    if ( self.terminalSynapses[ ii ].getWinnerCell()[ 0 ] == other.terminalSynapses[ ii ].getWinnerCell()[ 0 ]
                        and self.terminalSynapses[ ii ].getWinnerCell()[ 1 ] > lowerThreshold
                        and other.terminalSynapses[ ii ].getWinnerCell()[ 1 ] > lowerThreshold ):

                            terminalEquality += 1

        if incidentEquality >= equalityThreshold and terminalEquality >= equalityThreshold:
            return True
        else:
            return False

    def Inside( self, vector ):
    # Checks if given position is inside range.

        for i in range( self.dimensions ):
            if vector[ i ] < self.vector[ i ][ 0 ] or vector[ i ] > self.vector[ i ][ 1 ]:
                return False

        return True

    def CheckOverlap( self, FCellList, cellsPerColumn, synapsesToCheck, lowerThreshold ):
    # Check FColumnList for overlap with incidentSynapses. If overlap is above threshold return True, otherwise False.

        overlap = 0

        for synapse in synapsesToCheck:
            if FCellList[ synapse ]:
                overlap += 1

        if overlap >= self.activationThreshold:
            return True
        else:
            return False

    def AdjustThreshold( self, minActivation, maxActivation ):
    # Check the number of bundles that have selected winner. Use this to adjust activationThreshold.

        numWinners = 0

        for incB in self.incidentPermanences:
            if incB == 1.0:
                numWinners += 1

        self.activationThreshold = ( ( minActivation - maxActivation ) * np.exp( -1 * numWinners ) ) + maxActivation

    def DecayAndCreate( self, lastActiveCellsBool, permanenceIncrement, permanenceDecrement, initialPermanence, maxBundlesToAddPer, maxBundlesPerSegment, cellsPerColumn ):
    # Increase synapse strength to lastActive cells that already have synapses.
    # Decrease synapse strength to not lastActive cells that have synapses.
    # Build new synapses to lastActive cells that don't have synapses, but only one cell per column can have synapses.
    # Delete synapses whose permanences have decayed to zero.

        synapseToDelete = []

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
                thisColumn = int( cellIndex / cellsPerColumn )
                addNewSynapse = False

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
                    self.incidentSynapses.insert( synIndex, cellIndex )
                    self.incidentPermanences.insert( synIndex, initialPermanence )
                    synIndex += 1

        # Delete synapses whose permanences have decayed to zero.
        for toDel in synapseToDelete:
            del self.incidentSynapses[ toDel ]
            del self.incidentPermanences[ toDel ]

# ------------------------------------------------------------------------------

class FCell:

    def __init__( self, initialPermanence, numOCells, pctAllowedOCellConns, segmentDecay ):
    # Create a new feature level cell with synapses to OCells.

        self.active     = False
        self.lastActive = False
        self.predicted  = False
        self.winner     = False
        self.lastWinner = False

        self.numWinners = 0

        self.segments   = []
        self.segmentDecay = segmentDecay

        # Create synapse connections to OCells.
        numberOCellConns = int( pctAllowedOCellConns * numOCells )
        self.OCellConnections = sorted( sample( range( numOCells ), numberOCellConns ) )
        self.OCellPermanences = []
        for i in range( numberOCellConns ):
            self.OCellPermanences.append( uniform( initialPermanence - ( initialPermanence / 2 ), initialPermanence ) )

    def __repr__( self ):
    # Returns string properties of the FCell.

        toPrint = []
        for index in range( len( self.OCellConnections ) ):
            toPrint.append( ( self.OCellConnections[ index ], self.OCellPermanences[ index ] ) )

        return ( "< ( Connected OCells, Permanence ): %s >"
            % toPrint )

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

    def CheckIfPredicted( self, activeFCells, cellsPerColumn, vector, lowerThreshold ):
    # Checks if this cell has any segments predicting by checking incident overlap and vector.
    # If so then activate that segment and make cell predicted, otherwise not predicted.

# SHOULD IMPROVE THIS BY CHECKING IF MULTIPLE SEGMENTS ARE ACTIVE WE ONLY WANT THE ONE WITH THE MOST OVERLAP MAYBE.
# MAYBE COMBINE THE TWO INTO ONE, OR DELETE ONE, IF ACTIVE TOGETHER OFTEN.
        anySegments = False

        for segment in self.segments:
            if segment.Inside( vector ) and segment.CheckOverlap( activeFCells, cellsPerColumn, segment.incidentSynapses, lowerThreshold ):
                segment.active = True
                anySegments    = True

        if anySegments:
            self.predicted = True
        else:
            self.predicted = False

    def CheckIfSegsIdentical( self, segsToDelete ):
    # Compares all active segments to see if they have identical vectors and active synapse bundles.
    # If any do then delete one of them.

        if len( self.activeSegs ) > 1:
            for index1 in range( len( self.activeSegs ) - 1 ):
                for index2 in range( index1 + 1, len( self.activeSegs ) ):
                    actSeg1 = self.activeSegs[ index1 ]
                    actSeg2 = self.activeSegs[ index2 ]

                    if actSeg1 != actSeg2:
                        if self.segments[ actSeg1 ].Equality( self.segments[ actSeg2 ], self.equalityThreshold, self.lowerThreshold ):
                            NoRepeatInsort( segsToDelete, actSeg2 )
                    else:
                        print( "Error in CheckIfSegsIdentical()")
                        exit()

    def CreateSegment( self, vector, incidentCellWinners, initialPermanence, initialPosVariance, FActivationThresholdMin  ):
    # Create a new Segment, terminal on these active columns, incident on last active columns.

# THE PROBLEM WITH BELOW IS IF MORE THAN ONE CELL IN COLUMN IS ACTIVE (BECAUSE WE HYPOTHESIZE MULTIPLE FEATURES)
# THEN IT WILL CHOOSE THE ONE WITH THE LEAST TIMES WINNER TO BUILD SYNAPSE TO. BUT THIS WILL CROSS OVER FEATURE
# LINES AND MIX FEATURES. NOT WHAT WE WANT TO DO...

        newSegment = Segment( incidentCellWinners, initialPermanence, vector, initialPosVariance, FActivationThresholdMin )
        self.segments.append( newSegment )

    def SegmentLearning( self, lastVector, lastActiveFCellsBool, incidentCellWinners, initialPermanence, initialPosVariance,
        FActivationThresholdMin, FActivationThresholdMax, permanenceIncrement, permanenceDecrement,
        maxNewFToFSynapses, maxSynapsesPerSegment, cellsPerColumn ):
    # Perform learning on segments, and create new ones if neccessary.

        segsToDelete = []

        # Add time to all segments, and delete segments that haven't been active in a while.
        self.IncrementSegTime( segsToDelete )

        # If this cell is winner but the cell is not predicted then create a new segment to the lastActive FCells.
        if self.winner and not self.predicted:
            self.CreateSegment( lastVector, incidentCellWinners, initialPermanence, initialPosVariance, FActivationThresholdMin )

        # If there is more than one segment active check if they are idential, if so delete one.
#        self.CheckIfSegsIdentical( segsToDelete )

        # For every active and lastActive segment...
        for segment in self.segments:
            if segment.active:
                # If active segments have positive synapses to non-active columns then decay them.
                # If active segments do not have terminal or incident synapses to active columns create them.
                segment.DecayAndCreate( lastActiveFCellsBool, permanenceIncrement, permanenceDecrement,
                    initialPermanence, maxNewFToFSynapses, maxSynapsesPerSegment, cellsPerColumn )

                # Adjust the segments activation threshold depending on number of winners selected.
                segment.AdjustThreshold( FActivationThresholdMin, FActivationThresholdMax )

        # Delete segments that had all their incident or terminal synapses removed.
        self.DeleteSegments( segsToDelete )

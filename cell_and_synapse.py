from bisect import bisect_left
from random import uniform

def BinarySearch( list, val ):
# Search a sorted list: return False if val not in list, and True if it is.

    i = bisect_left( list, val )

    if i != len( list ) and list[ i ] == val:
        return True

    else:
        return False

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

class OSegment:

    def __init__( self, OCellList, terminalFCellList, terminalColumn, incidentFCellList, initialPermanence, vector, posRange ):
    # Create a new OSegment, which connects multiple OCells to multiple FCells in same column. One will be chosen
    # as winner eventually.
    # Within this create an FSegment, with multiple synapses to FCells from these FCells.
    # Also a vector which will be used to activate segment.

        self.active          = False
        self.lastActive      = False
        self.primed          = True
        self.timeSinceActive = 0

        # OSegment portion--------
        self.OCells          = OCellList
        self.terminalColumn  = terminalColumn
        self.terminalFCells  = terminalFCellList
        self.OToFPermanences = []
        for i in range( len( terminalFCellList ) ):
            self.OToFPermanences.append( uniform( initialPermanence - ( initialPermanence / 2 ), initialPermanence ) )
        # FSegment postion--------
        self.FToFSynapses    = incidentFCellList
        self.FToFPermanences = []
        for i in range( len( incidentFCellList ) ):
            self.FToFPermanences.append( uniform( initialPermanence - ( initialPermanence / 2 ), initialPermanence ) )

        # Vector portion---------
        self.dimensions      = len( vector )      # Number of dimensions vector positions will be in.
        self.vector          = []                 # A list of tuples, one for each dimension, with a range for location.
        for i in range( self.dimensions ):
            posI = ( vector[ i ] - posRange / 2, vector[ i ] + posRange / 2 )
            self.vector.append( posI )

    def __lt__(self, other):
    # OSegment1 < OSegment1 calls OSegment1.__lt__( OSegment2 )

        if type( other ) == type( self ):
            return self.terminalColumn < other.terminalColumn

        else:
            return self.terminalColumn < other

    def __eq__(self, other):
    # OSegment1 == OSegment1 calls OSegment1.__eq__( OSegment2 )

        if type( other ) == type( self ):
            return self.terminalColumn == other.terminalColumn

        else:
            return self.terminalColumn == other

    def __repr__(self):

            return ( "< Terminal Column: %s, Vector: %s, \n Active: %s, Last Active: %s, Primed: %s, Time Since Active: %s \n Terminal Cells: %s, \n Incident Cells: %s >"
                % (self.terminalColumn, self.vector, self.active, self.lastActive, self.primed, self.timeSinceActive, self.terminalFCells, self.FToFSynapses ) )

    def Inside( self, vector ):
    # Checks if given position is inside range.

        for i in range( self.dimensions ):
            if vector[ i ] < self.vector[ i ][ 0 ] or vector[ i ] > self.vector[ i ][ 1 ]:
                return False

        return True

    def OCellOverlap( self, OCellList, overlapThreshold ):
    # Check OCellList for overlap with self.OCells. If overlap is above threshold return True, otherwise False.

        overlap = 0

        for oCell in self.OCells:
            if BinarySearch( OCellList, oCell ):
                overlap += 1

        if overlap >= overlapThreshold:
            return True
        else:
            return False

    def FCellOverlap( self, FCellList, overlapThreshold ):
    # Check FCellList for overlap with self.FToFSynapses. If overlap is above threshold return True, otherwise False.

        overlap = 0

        for fCell in self.FToFSynapses:
            if BinarySearch( FCellList, fCell ):
                overlap += 1

        if overlap >= overlapThreshold:
            return True
        else:
            return False


class FCell:

    def __init__( self, ID ):
    # Create a new inactive feature level cell with no synapses.

        self.ID             = ID
        self.active         = False     # Means column burst, or cell was predictive and then column fired.
        self.predictive     = False     # Means synapses on connected segments above activationThreshold.
        self.terminalWinner = 0         # Number of OSegments this cell is chosen as the terminal FCell winner for.

class OCell:

    def __init__( self, ID ):
    # Create a new inactive feature level cell with no synapses.

        self.ID             = ID
        self.active         = False
